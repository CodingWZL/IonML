from pymatgen import Structure, Composition, Element
from matminer.featurizers.structure import GlobalSymmetryFeatures
from matminer.featurizers.site import AverageBondLength, AverageBondAngle
from pymatgen.analysis.local_env import VoronoiNN
from matminer.featurizers.composition import ElementFraction, BandCenter
from xgboost.sklearn import XGBClassifier
import numpy as np
import random

def structure_symmetry(structure):
    """
    :param structure: Pymatgen Structure
    :return: space_group, crystal_system_int, is_centrosymmetric, n_symmetry_ops
    crystal_system_int:
        "triclinic": 7,
        "monoclinic": 6,
        "orthorhombic": 5,
        "tetragonal": 4,
        "trigonal": 3,
        "hexagonal": 2,
        "cubic": 1,
    """
    GSF = GlobalSymmetryFeatures()
    space_group, crystal_system, crystal_system_int, is_centrosymmetric, n_symmetry_ops = GSF.featurize(structure)
    if is_centrosymmetric is True:  # convert the boolean to int
        is_centrosymmetric = 1
    else:
        is_centrosymmetric = 0
    a = [0,0,0,0,0,0,0]
    if crystal_system_int == "cubic":
        a[0] = 1
    elif crystal_system_int == "hexagnoal":
        a[1] = 1
    elif crystal_system_int == "trigonal":
        a[2] = 1
    elif crystal_system_int == "tetragonal":
        a[3] = 1
    elif crystal_system_int == "orthorhombic":
        a[4] = 1
    elif crystal_system_int == "monoclinic":
        a[5] = 1
    elif crystal_system_int == "triclinic":
        a[6] = 1
    return a, is_centrosymmetric

# Calculate the average atomic volume and average volume atoms for a crystal structure
def average_atomic_volume(structure):
    volume = structure.volume
    number_atoms = structure.num_sites
    average_atom_volume = number_atoms/volume
    average_volume_atom = volume/number_atoms
    return average_atom_volume, average_volume_atom


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


# calculate the fraction of Li of a composition
def Li_fraction(composition):
    """
    :param structure: a string of composition
    :return: one dimensional matrix of atomic number and fraction
    """
    composition = Composition(composition)
    symbol = composition.elements
    dic_metal={}
    dic_nonmetal={}
    for i in range(len(symbol)):
        if symbol[i].is_metal:
            dic_metal[str(symbol[i])]=int(symbol[i].number)
        else:
            dic_nonmetal[str(symbol[i])]=int(symbol[i].number)
    #print()
    dic_metal = sorted(dic_metal.items(), key=lambda kv:kv[1])
    dic_nonmetal = sorted(dic_nonmetal.items(), key=lambda kv:kv[1])
    dic = dic_metal+dic_nonmetal
    #print(dic[0][0])
    symbol=[]
    for i in range(len(dic)):
        a = Element(str(dic[i][0]))
        symbol.append(a)
    #print(symbol)
    EF = ElementFraction()
    site_feature = np.zeros(12)
    index = 0
    for i in range(len(symbol)):
        if index > 0:
            if not symbol[i].is_metal and index <= 6:
                index = 6 # start index of non_metal
        site_feature[index] = symbol[i].number
        site_feature[index+1] = EF.featurize(composition)[symbol[i].number-1]
        index += 2
    return site_feature


# calculate the band center by first ionization energy and electron affinity
def band_center(composition):
    composition = Composition(composition)
    BC = BandCenter()
    bandcenter = BC.featurize(composition)
    return bandcenter

def predict(feature):
    enreg = XGBClassifier()
    enreg.load_model('xgb-1.model')
    p_1 = enreg.predict_proba(feature)
    enreg.load_model('xgb-2.model')
    p_2 = enreg.predict_proba(feature)
    enreg.load_model('xgb-3.model')
    p_3 = enreg.predict_proba(feature)
    enreg.load_model('xgb-4.model')
    p_4 = enreg.predict_proba(feature)
    enreg.load_model('xgb-5.model')
    p_5 = enreg.predict_proba(feature)
    p = (p_1 + p_2 + p_3 + p_4 + p_5) / 5
    return p


def main():
    features = []
    structure = Structure.from_file("LiBF-CONTCAR.cif")
    ###### Structure Feature ######
    crystal_system, centrosymmetric = structure_symmetry(structure)
    average_atom_volume, average_volume_atom = average_atomic_volume(structure) # dimension:2

    ###### Site Feature ######
    neighbor_num = average_neighbor_num(structure) # dimension:1
    bond_length = average_bond_length(structure) # dimension:1
    bond_angle = average_bond_angle(structure) # dimension:1
    effective_ratio = effective_location(structure) # dimension:1

    ###### Component Feature ######
    a = structure.composition
    site_feature = list(Li_fraction(Composition(a))) # dimension:12
    bandcenter = band_center(Composition(a)) # dimension:1

    feature = crystal_system
    feature.append(centrosymmetric)
    feature.append(average_atom_volume)
    feature.append(average_volume_atom)
    feature.append(neighbor_num)
    feature.append(bond_length)
    feature.append(bond_angle)
    feature.append(effective_ratio)
    feature = feature + site_feature + bandcenter

    features = features + feature
    features = np.array(features)
    features = features.reshape(-1,27)
    #np.savetxt("features.txt", features)
    result = predict(features)[0]
    print(result)

if __name__ == '__main__':
    main()
