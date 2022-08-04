import os
#from matminer.featurizers.structure.sites import SiteStatsFingerprint
from pymatgen import Lattice, Structure, Composition
from StructureFeature import structure_symmetry, average_atomic_volume
from SiteFeature import average_neighbor_num, average_bond_length, average_bond_angle, angular_fourier_series, effective_location
from ComponentFeature import Li_fraction, band_center
import numpy as np

# Citrination API Key: K7680bck0TbSG5uyRtnzDwtt
# Materials project API Key: uTvDJ6mH2k9aXKQXJ6
# os.environ['CITRINE_KEY'] = ":Q4lUr9HO38ZAJNr21lbljgtt"

#cif_number = np.loadtxt("log.txt")

def main():
    #f = open("component.txt", "r")
    features = []
    for i in range(1,2):
        structure = Structure.from_file("LiCaBO-CONTCAR.cif")
        ###### Structure Feature ######
        crystal_system, centrosymmetric = structure_symmetry(structure)
        print(crystal_system, centrosymmetric)
        #
        average_atom_volume, average_volume_atom = average_atomic_volume(structure) # dimension:2

        ###### Site Feature ######
        neighbor_num = average_neighbor_num(structure) # dimension:1
        bond_length = average_bond_length(structure) # dimension:1
        bond_angle = average_bond_angle(structure) # dimension:1
        effective_ratio = effective_location(structure) # dimension:1

        ###### Component Feature ######
        a = structure.composition
        #print(a)
        site_feature = list(Li_fraction(Composition(a))) # dimension:12
        bandcenter = band_center(Composition(a)) # dimension:1

        feature = crystal_system# + site_feature + bandcenter
        feature.append(centrosymmetric)
        feature.append(average_atom_volume)
        feature.append(average_volume_atom)
        feature.append(neighbor_num)
        feature.append(bond_length)
        feature.append(bond_angle)
        feature.append(effective_ratio)
        feature = feature + site_feature + bandcenter

        features = features + feature
        print(i)
    features = np.array(features)
    features = features.reshape(-1,27)
    np.savetxt("features-Li.txt", features)

if __name__ == '__main__':
    main()
