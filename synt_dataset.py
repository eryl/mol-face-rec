"""
This is from ChemPixCH 
https://github.com/mtzgroup/ChemPixCH/blob/main/data/synthetic/pipeline_stages/build.py

This file reads a file of smiles strings and generates a 
hand-drawn chemical structure dataset of these molecules.
1. Collect smiles strings from txt file
2. Collect background images
3. For each smiles string:
    3a. Convert smiles string to ong of molecule
    3b. Augment molecule image using molecule 
        augmentation pipeline
    3c. Randomly select background image
    3d. Augment background image using background
        augmentation pipeline
    3e. Combine augmented molecule and augmented background
        using random weighted addition
    3f. Degrade total image
    3g. Save image to folder 
"""

import math
import os
import glob
import numpy as np
import random 
from pathlib import Path
from copy import copy, deepcopy
from dataclasses import dataclass, field, asdict
import multiprocessing

from multiprocessing import Pool
from typing import Union, Optional, Mapping, Sequence, Tuple, List
from collections.abc import Collection
import types

import rdkit 
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from scipy.sparse import data

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

import torchvision.transforms as tvt
import torchvision.transforms.functional as tvf

import kornia.augmentation as kaug
from kornia.morphology import dilation, erosion

from PIL import Image
from torchvision.transforms.transforms import RandomResizedCrop

from tqdm import trange, tqdm


from rdkit.Chem.Draw import rdMolDraw2D


try:
    @profile
    def foo():
        pass
    del foo

except NameError:
    def profile(f):
        return f


class RandomVariableInstantiationError(Exception):
    def __init__(self, obj, message):
        self.obj = obj
        self.message = message

    def __str__(self):
        return f"{self.message}: {self.obj}"


class RandomVariable:
    def draw_value(self, rng):
        raise NotImplementedError('draw_value is not implemented for RandomVariable')

class RandomUniform(RandomVariable):
    def __init__(self, low, high) -> None:
        super().__init__()
        self.low = low
        self.high = high

    def draw_value(self, rng: np.random.Generator):
        return float(rng.uniform(self.low, self.high))

class RandomInteger(RandomVariable):
    def __init__(self, low, high) -> None:
        super().__init__()
        self.low = low
        self.high = high

    def draw_value(self, rng: np.random.Generator):
        return int(rng.integers(self.low, self.high))

class RandomCategorical(RandomVariable):
    def __init__(self, values, p=None) -> None:
        self.values = values
        self.p = p
    
    def draw_value(self, rng: np.random.Generator):
        return rng.choice(self.values, p=self.p)


class RandomBool(RandomVariable):
    def __init__(self, p=0.5) -> None:
        super().__init__()
        self.p = p

    def draw_value(self, rng: np.random.Generator):
        return rng.random() < self.p


def instantiate_random_vars(obj, rng: np.random.Generator):
    """Make a nested traversal of obj and replace any RandomVariable with a value drawn from its distribution"""
    non_collection_types = (str, bytes, bytearray, np.ndarray)
    try:
        if isinstance(obj, (type, types.FunctionType, types.LambdaType, types.ModuleType)):
            return obj
        if isinstance(obj, RandomVariable):
            return obj.draw_value(rng)
        elif isinstance(obj, Mapping):
            return type(obj)({k: instantiate_random_vars(v, rng) for k, v in obj.items()})
        elif isinstance(obj, Collection) and not isinstance(obj, non_collection_types):
            return type(obj)(instantiate_random_vars(x, rng) for x in obj)
        elif hasattr(obj, '__dict__'):
            try:
                obj_copy = copy(obj)
                obj_copy.__dict__ = instantiate_random_vars(obj.__dict__, rng)
                return obj_copy
            except TypeError:
                return obj
        else:
            return obj
    except TypeError as e:
        raise RandomVariableInstantiationError(obj, "Failed to materialize") from e



FloatOption = Union[RandomUniform, float]
BoolOption = Union[RandomBool, bool]
IntOption = Union[RandomInteger, int]

@dataclass
class DrawingOptions:
    addAtomIndices: bool = False  #adds atom indices to drawings. Default False.
    addBondIndices: bool = False  #adds bond indices to drawings. Default False.
    addStereoAnnotation: bool = field(default_factory=RandomBool)  #adds R/S and E/Z to drawings. Default False.
    additionalAtomLabelPadding: FloatOption = field(default_factory=lambda: RandomUniform(1, 10))  #additional padding to leave around atom labels. Expressed as a fraction of the font size.
    annotationFontScale: FloatOption = 0.75  #Scale of font for atom and bond annotation relative to atomlabel font. Default=0.75.
    atomHighlightsAreCircles: bool = False  #forces atom highlights always to be circles.Default (false) is to put ellipses roundlonger labels.
    atomLabelDeuteriumTritium: bool = False  #labels deuterium as D and tritium as T
    bondLineWidth: IntOption = field(default_factory=lambda: RandomInteger(1,10)) #if positive, this overrides the default line width for bonds
    circleAtoms: bool = True
    comicMode: BoolOption = field(default_factory=RandomBool)  #simulate hand-drawn lines for bonds. When combined with a font like Comic-Sans or Comic-Neue, this gives xkcd-like drawings. Default is false.
    continuousHighlight: bool = True
    explicitMethyl: BoolOption = False  #Draw terminal methyls explictly. Default is false.
    fixedBondLength: FloatOption = -1  #If > 0.0, fixes bond length to this number of pixelsunless that would make it too big. Default -1.0 meansno fix. If both set, fixedScale takes precedence.
    fixedScale: FloatOption = -1  #If > 0.0, fixes scale to that fraction of width ofdraw window. Default -1.0 means adjust scale to fit.
    fontFile: Optional[str] = None  #Font file for use with FreeType text drawer
    maxFontSize: int = 40  #maximum font size in pixels. default=40, -1 means no maximum.
    minFontSize: int = 12  #minimum font size in pixels. default=12, -1 means no minimum.
    rotate: FloatOption = field(default_factory=lambda: RandomUniform(-180, 180))
    multipleBondOffset: FloatOption = 0.15  #offset (in Angstroms) for the extra lines in a multiple bond
    scaleBondWidth: bool = False  #Scales the width of drawn bonds using image scaling.

    def get_options(self):
        return asdict(self)



@dataclass 
class MolAugmentOptions:
    resize_p: float = 0
    blur_p: float = 0
    blur_size: int = 5
    blur_sigma: float = 1
    blur_strength: float = 1
    affine_p: float = 0.9
    affine_angle: FloatOption = 0
    affine_translate: Tuple[FloatOption, FloatOption] = (0,0)
    flip_h_p: float = 0.5
    flip_v_p: float = 0.5
    erode_p: float = 0
    dilate_p: float = 0
    aspect_ratio_p: float = 0
    distort_p: float = 0
    contrast_p: float = 0.4
    contrast_factor: FloatOption = 1
    gamma_p: float = 0
    gamma_gamma: float = 1
    gamma_gain: float = 1
    sharpness_p: float = 0
    sharpness_factor: float = 1

    # translate_mol    
    #img = translate_mol(img, rng)




@dataclass
class SyntheticMoldrawingDatasetConfig:
    width: int   # Final output width
    height: int  # Final output heigth

    working_size: int = 256  # The size images will be resized to and used during augmentation

    rdkit_size: IntOption = field(default_factory=lambda: RandomInteger(80, 800))
    rdkit_kekulize: BoolOption = field(default_factory=RandomBool)
    rdkit_wedgeBonds: BoolOption = field(default_factory=RandomBool)

    drawing_options: DrawingOptions = field(default_factory=DrawingOptions)
    mol_augmentation_options: MolAugmentOptions = field(default_factory=MolAugmentOptions)

    def get_options(self, rng) -> 'SyntheticMoldrawingDatasetConfig':
        '''Return a copy of this SyntheticMoldrawingDatasetConfig, where any random variable is replaced by a value drawn from its distribution'''
        options = deepcopy(self)
        instantiated_options = instantiate_random_vars(options, rng)
        return instantiated_options


class SyntheticMoldrawingDataset:
    def __init__(self, *, config: SyntheticMoldrawingDatasetConfig, smiles_list, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng
        self.smiles_list = smiles_list
        self.config = config
    
    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, item):
        smiles = self.smiles_list[item]
        img1 = self.smiles_to_synthetic(smiles)
        img2 = self.smiles_to_synthetic(smiles)
        return img1, img2

    def smiles_to_rdkitmod(self, s, instantiated_options: SyntheticMoldrawingDatasetConfig):
        '''Generate RDKit image from smiles string'''
        m = Chem.MolFromSmiles(s)
        dopts = rdMolDraw2D.MolDrawOptions()
        dopts.prepareMolsForDrawing = True
        drawing_options = instantiated_options.drawing_options
        for k,v in drawing_options.get_options().items():
            if v is not None:
                setattr(dopts, k, v)
        draw_size = instantiated_options.rdkit_size
        kekulize = instantiated_options.rdkit_kekulize
        wedgeBonds = instantiated_options.rdkit_wedgeBonds

        img = rdkit.Chem.Draw.MolToImage(m, size=(draw_size, draw_size), kekulize=kekulize, wedgeBonds=wedgeBonds, options=dopts)
        grey_img = img.convert('L').convert('RGB')
        img = tvf.to_tensor(grey_img)
        resized_img = tvf.resize(img, [instantiated_options.working_size, instantiated_options.working_size])
        return resized_img

    @profile
    def smiles_to_synthetic(self, s):
        instantiated_options = self.config.get_options(self.rng)
        mol = self.smiles_to_rdkitmod(s, instantiated_options)
        return mol
        


def get_smiles(filename):
    ''' Read smiles data from *.formulas.txt file'''
    with open(filename) as f:
        lines = f.readlines()
    smiles = [s.split()[0] for s in lines]
    return smiles


def get_background_imgs(path: Path):
    '''Reads in background dataset'''
    bkg_files = path.glob('*.png')
    bkgs = torch.stack([tvf.to_tensor(Image.open(b).convert('RGB')).squeeze() for b in bkg_files])
    return bkgs


class RandomDilation(torch.nn.Module):
    def __init__(self, *, p: float, kernel_size_low: int, kernel_size_high: int) -> None:
        super().__init__()
        self.p = p
        self.kernel_size_low = kernel_size_low
        self.kernel_size_high = kernel_size_high
         # We only use odd kernel sizes
        self.kernels = [torch.ones((d,d), requires_grad=False) for d in range(self.kernel_size_low, self.kernel_size_high+1, 2)]

    def forward(self, image):
        if torch.rand(1).item() < self.p:
            kernel_i = torch.randint(len(self.kernels), (1,))
            kernel = self.kernels[kernel_i]
            dilated = dilation(image, kernel)
            return dilated
        else:
            return image


class RandomErosion(torch.nn.Module):
    def __init__(self, *, p: float, kernel_size_low: int, kernel_size_high: int) -> None:
        super().__init__()
        self.p = p
        self.kernel_size_low = kernel_size_low
        self.kernel_size_high = kernel_size_high
        # We only use odd kernel sizes
        self.kernels = [torch.ones((d,d), requires_grad=False) for d in range(self.kernel_size_low, self.kernel_size_high+1, 2)]

    def forward(self, image):
        if torch.rand(1).item() < self.p:
            kernel_i = torch.randint(len(self.kernels), (1,))
            kernel = self.kernels[kernel_i]
            eroded = erosion(image, kernel)
            return eroded
        else:
            return image


class RandomFill(torch.nn.Module):
    def __init__(self, *, p: float, fill_p: float, fill_value: float = 1., separate_channels: bool=False) -> None:
        '''Randomly fill image with desired fill value.
        :param p: probability of applying this transormation
        :param fill_p: probabilitiy of filling a pixel with the fill value
        :param fill_value: the value to fill with
        '''
        super().__init__()  
        self.p = p
        self.fill_p = fill_p
        self.fill_value = fill_value
        self.separate_channels = separate_channels

    def forward(self, image):
        *leading_dims, n_channels, height, width = image.shape
        with torch.no_grad():
            if self.separate_channels:
                bernoulli_p = torch.full_like(image, self.fill_p)
                fill_mask = torch.bernoulli(bernoulli_p)
            else:
                bernoulli_p = torch.full((*leading_dims, 1, height, width), self.fill_p)
                tile_dims = (*(1,)*len(leading_dims), n_channels, 1, 1)  # only tile the channels axis
                fill_mask = torch.bernoulli(bernoulli_p).tile(tile_dims)
        # We wan't to randomly apply the fill to the batch examples independently, 
        # we achieve this by making a second pass over the fill mask.
        # We make the mask have ones in the image dimension axises, 
        # so we can broadcast the result using multiplication
        batch_bernoulli_p = torch.full((*leading_dims, 1, 1, 1), self.p)
        batch_mask = torch.bernoulli(batch_bernoulli_p)
        fill_mask = fill_mask * batch_mask
        filled = image.masked_fill(fill_mask.to(torch.bool), self.fill_value)
        return filled
        

class MolImageAugmenter:
    def __init__(self, *, working_size, backgrounds) -> None:
        self.aug = torch.nn.Sequential(
            RandomFill(p=0.5, fill_p=0.5, fill_value=1.),
            RandomErosion(p=0.2, kernel_size_low=3, kernel_size_high=7),
            RandomDilation(p=0.2, kernel_size_low=3, kernel_size_high=7),
            tvt.RandomPerspective(p=0.5, distortion_scale=0.5, fill=1),
            tvt.RandomApply([tvt.RandomAffine((-180, 180), translate=(0.1, 0.1), scale=(0.7,1.3), fill=1)], p=1),
            tvt.RandomApply([tvt.RandomResizedCrop(working_size, scale=(0.5, 1))], p=1,),
            tvt.Resize((224,224)))
        self.rng = np.random.default_rng()
        self.backgrounds = backgrounds

    def __call__(self, batch):
        auged_batch = self.aug(batch)
        background_indices = self.rng.choice(len(self.backgrounds), replace=True, size=auged_batch.shape[0])
        chosen_backgrounds = self.backgrounds[background_indices]
        augmented_backgrounds = self.aug(chosen_backgrounds)
        mix_coefficient = 0.7
        mixed_batch = mix_coefficient*auged_batch + (1-mix_coefficient)*augmented_backgrounds
        return mixed_batch


class MolImageCollator:
    def __init__(self, *, augmenter: MolImageAugmenter) -> None:
        self.augmenter = augmenter

    def __call__(self, batch):
        stacked_batch = torch.cat([torch.stack(pair) for pair in batch], dim=0)
        auged_batch = self.augmenter(stacked_batch)
        n_images = auged_batch.shape[0]
        n_pairs = n_images // 2
        pairwise_images = auged_batch.view(n_pairs, 2, *auged_batch.shape[1:])
        return pairwise_images


def augmenter_worker(results_queue: multiprocessing.Queue, 
                     dataset: SyntheticMoldrawingDataset, 
                     augmenter: MolImageAugmenter, 
                     output_dir: Path, 
                     indices: List[int], 
                     n_pairs: int):
    for i in indices:
        img_dir = output_dir / str(i) 
        img_dir.mkdir(exist_ok=True)
        for j in range(n_pairs):
            mol_a, mol_b = dataset[i]

            augmented_a = augmenter(mol_a.unsqueeze(0))
            img_a = tvf.to_pil_image(augmented_a.squeeze())
            img_filename = img_dir / f'{j*2}.png'
            img_a.save(img_filename, format='png')

            augmented_b = augmenter(mol_b.unsqueeze(0))
            img_b = tvf.to_pil_image(augmented_b.squeeze())
            img_filename = output_dir / str(i) / f'{j*2+1}.png'
            img_b.save(img_filename, format='png')
        results_queue.put(i)


def main():
    from argparse import ArgumentParser

    # Collect background images
    parser = ArgumentParser(description='Script for training molecular face recognition')
    parser.add_argument('training_smiles', help='Path to a file listing SMILES to use for training', type=Path)
    parser.add_argument('dev_smiles', help='Path to file listing SMILES to use for early stopping', type=Path)
    parser.add_argument('test_smiles', help='Path to file listing  SMILES to use for testing', type=Path)
    parser.add_argument('--backgrounds', help='Path to directory containing images to use for the background', type=Path)
    parser.add_argument('--random-seed', help='Constant to seed the random number generator with', default=1729)
    parser.add_argument('--output-dir', help='Where to write augmented images', type=Path, default=Path('augmented_images'))
    args = parser.parse_args()

    with open(args.training_smiles) as fp:
        training_smiles = [line.strip() for line in fp]
    with open(args.dev_smiles) as fp:
        dev_smiles = [line.strip() for line in fp]
    with open(args.test_smiles) as fp:
        test_smiles = [line.strip() for line in fp]

    background_images = get_background_imgs(args.backgrounds)

    dataset_rng = np.random.default_rng(args.random_seed)
    batch_size = 4
    dataset_config = SyntheticMoldrawingDatasetConfig(width=224, height=224)
    augmenter = MolImageAugmenter(backgrounds=background_images, working_size=dataset_config.working_size)

    output_dir = args.output_dir / 'train'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    dataset = SyntheticMoldrawingDataset(config=dataset_config, smiles_list=training_smiles)
    n = len(dataset)
    
    n_pairs = 5
    n_processes = 12
    indices = list(range(n))
    indices_per_process = int(math.ceil(n / n_processes))

    results_queue = multiprocessing.Queue()
    processes = []
    for i in range(n_processes):
        process_indices = indices[i*indices_per_process: (i+1)*indices_per_process]
        p = multiprocessing.Process(target=augmenter_worker, args=(results_queue, dataset, augmenter, output_dir, process_indices, n_pairs))
        processes.append(p)
        p.start()

    indices = set(indices)
    with trange(n, desc='Training pairs') as pbar:
        while indices:
            i = results_queue.get()
            indices.remove(i)
            pbar.update()
            

    for dataset_name, dataset_smiles in [('dev', dev_smiles), ('test', test_smiles)]:
        output_dir = args.output_dir / dataset_name
        output_dir.mkdir(exist_ok=True, parents=True)

        dataset = SyntheticMoldrawingDataset(config=dataset_config, smiles_list=dataset_smiles)
        n = len(dataset)

        for i in trange(n, desc=f"{dataset_name}"):
            img_dir = output_dir / str(i) 
            img_dir.mkdir(exist_ok=True)
        
            mol_a, mol_b = dataset[i]

            mol_a = tvf.resize(mol_a, [dataset_config.width, dataset_config.height])
            img_a = tvf.to_pil_image(mol_a)
            img_filename = img_dir / f'{j*2}.png'
            img_a.save(img_filename, format='png')

            mol_b = tvf.resize(mol_b, [dataset_config.width, dataset_config.height])
            img_b = tvf.to_pil_image(mol_b)
            img_filename = output_dir / str(i) / f'{j*2+1}.png'
            img_b.save(img_filename, format='png')
            

if __name__ == "__main__":
    main()
    