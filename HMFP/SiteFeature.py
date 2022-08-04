from pymatgen import Lattice, Structure, Element
from pymatgen.core import Composition
#from matminer.featurizers.composition import ElementFraction, BandCenter
from matminer.featurizers.site import AverageBondLength, AverageBondAngle
from matminer.featurizers.site import AngularFourierSeries
from pymatgen.analysis.local_env import VoronoiNN, CrystalNN
import numpy as np
import matplotlib.pyplot as plt
import random

# Automated close-loop optimization and design of superionic conductors


# calculate the average length for all Li sites
def average_bond_length(structure):
    """
    :param structure: a Pymatgen Structure
    :return: average length between neighbor atoms for all Li sites
    """
    ABL = AverageBondLength(VoronoiNN(cutoff=5.0))
    index = structure.indices_from_symbol("Li")
    #features = ABL.featurize(structure, 0)
    bond_length = np.mean([ABL.featurize(structure, i) for i in index])
    return bond_length


# calculate the average angle for all Li sites
def average_bond_angle(structure):
    """
    :param structure: a Pymatgen Structure
    :return: average angle between neighbor atoms for all Li sites
    """
    ABA = AverageBondAngle(VoronoiNN(cutoff=5.0))
    index = structure.indices_from_symbol("Li")
    #features = ABA.featurize(structure, 0)
    bond_angle = np.mean([ABA.featurize(structure, i) for i in index])
    return bond_angle


# calculate the average number of neighbors for all Li sites
def average_neighbor_num(structure):
    """
    :param structure: a Pymatgen Structure
    :return: average number of neighbors for all Li sites
    """
    sites = [site for site in structure.sites if site.specie.symbol == "Li"]
    neighbor_num = [len(structure.get_neighbors(site, r=5.0)) for site in sites]
    neighbor_num = np.mean(neighbor_num)
    return neighbor_num


# calculate the average fourier_series for all Li sites
def angular_fourier_series(structure, show=0):
    """
    :param structure: a Pymatgen Structure
    :param show: whether to show the fourier_series
    :return: average fourier_series for all Li sites, dimension = (cutoff/width)**2
    """
    AFS = AngularFourierSeries.from_preset(preset="gaussian", width=0.5, cutoff=5.0)
    index = structure.indices_from_symbol("Li")
    angular_fourier_series = []
    for i in index:
        angular_fourier_series.append(AFS.featurize(structure, i))
    average_angular_fourier_series = np.mean(angular_fourier_series, axis=0)  # average on columes
    if show == 1:
        plt.figure(1)
        x = np.arange(100)  # dimension = (5.0/0.5)**2 = 100
        plt.bar(x, angular_fourier_series)
        plt.show(average_angular_fourier_series)
    return average_angular_fourier_series


# calculate the effective location for Li
def effective_location(structure):
    """
    :param structure: a Pymatgen Structure
    :return: ratio effective location for Li
    """
    [a, b, c] = structure.lattice.abc
    all_sites = structure.sites
    count = 0
    for i in range(0, 1000): # 1000 random points
        x = random.randint(0, int(a*100))/100
        y = random.randint(0, int(b*100))/100
        z = random.randint(0, int(c*100))/100
        count += 1
        for j in range(0, len(all_sites)):
            specie = all_sites[j].specie    # element specie for a site
            [x_site, y_site, z_site] = all_sites[j].coords  # coordination of a site
            distance = np.sqrt((x-x_site)**2 + (y-y_site)**2 + (z-z_site)**2) # distance between a random point and a site
            if distance < specie.average_ionic_radius + Element.Li.average_ionic_radius + 0.25:  # Li: 0.9
                count -= 1
                break
    effective_ratio = count/1000
    return effective_ratio



def main():
    structure = Structure.from_file('Dy2HfS5_mp-1198001_computed.cif')
    structure = Structure.from_file('Li7P3S11_mp-641703_primitive.cif')
    #print(structure.lattice.abc)

    print(effective_location(structure))

if __name__ == '__main__':
    main()
