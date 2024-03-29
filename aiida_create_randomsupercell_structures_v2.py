#!/usr/bin/env python
import aiida
aiida.load_profile()
import os
from aiida.orm import Group
from aiida_create_solutesupercell_structures import *
import ase
import ase.build
import click
import numpy as np
import random

def determine_selection(concentrations):
    if not np.allclose(np.sum(concentrations), 1):
        raise Exception("concentrations:{} do not sum to 1".format(concentrations))
    random = np.random.rand()
    cumm_concentration = np.cumsum(concentrations)
    for i in range(len(cumm_concentration)):
        if random <= cumm_concentration[i]:
            return i
    raise Exception("Error: no selection chosen!")

def randomize_asestructure(ase_structure, elements, concentrations, seed):
    random_f = np.random.RandomState(seed).rand()
    for i in range(len(ase_structure)):
        element_index = determine_selection(concentrations)
        element_toinsert = elements[element_index]
        if element_toinsert == "Vac":
            print("DELETED!!!")
            del ase_structure[i]
        else:
            ase_structure[i].symbol = element_toinsert
    return ase_structure

def get_averaged_lattice(lattices, concentrations):
    if len(lattices) != len(concentrations):
        raise Exception("Number of lattices must match concentrations")
    lattice_concentration = list(zip(lattices, concentrations))
    lattice_normalization = np.sum([x[1] for x in 
                                    lattice_concentration
                                    if x[0] >= 0 ])
    average_lattice = np.sum([x[0]*x[1]/lattice_normalization for x in 
                              lattice_concentration
                              if x[0] >= 0])
    return average_lattice

@click.command()
@click.option('-me', '--matrix_elements', required=True,
              help="list of elements to be used in the matrix")
@click.option('-a', '--lattice_sizes', required=True,
              help="list of lattice sizes (in Ang) to use in the same order as the elements "
                   "the system will compute an average based on concentration "
                   "-1 indicates the element will not be included in the average")
@click.option('-c', '--concentrations', required=True,
              help="list of concentrations to use in the same order as the elements"
                   "concentrations must sum to 1")
@click.option('-rdisp', '--random_displacement', required=True, type=float,
              help="stdev of random displacement in ang")
@click.option('-ns', '--number_samples', required=True, type=int,
              help="Number of samples to generate")
@click.option('-spsh', '--supercell_shape', required=True,
              help="shape of the supercell to use, format: Nx,Ny,Nz")
@click.option('-sg', '--structure_group_label', required=True,
              help="Output AiiDA group to store created structures")
@click.option('-sgd', '--structure_group_description', default="",
              help="Description for output AiiDA group")
@click.option('-nost', '--nostore', is_flag=True,
              help="Do not store just dump out file directly")
@click.option('-dr', '--dryrun', is_flag=True,
              help="Prints structures and extras but does not store anything")
def launch(matrix_elements, lattice_sizes, concentrations,
           random_displacement, number_samples, supercell_shape,
           structure_group_label, structure_group_description,
           nostore,
           dryrun):
    """
    Script for generating random FCC supercells, where the matrix elements 
    """
    if not dryrun:
        structure_group = Group.objects.get_or_create(
                             label=structure_group_label, description=structure_group_description)[0]
    else:
        structure_group = None

    if nostore:
        dumpdir = os.path.abspath("./RANDOM_DUMP")
        print("STORING RESULTS IN ", dumpdir)
        os.makedirs(dumpdir, exist_ok=True)

    supercell_shape = supercell_shape.split(',')
    if len(supercell_shape) != 3:
        sys.exit("supercell_shape must be of the form Nx,Ny,Nz")
    matrix_elements = prep_elementlist(matrix_elements)
    lattice_sizes = [float(x) for x in lattice_sizes.split(',')]
    concentrations = [float(x) for x in concentrations.split(',')]

    matching_lists = [matrix_elements, lattice_sizes, concentrations]
    if not all(len(x) == len(matching_lists[0]) for x in matching_lists):
        raise Exception("unequal matrix_elements, lattice_sizes or concentrations")



    for i in range(number_samples):
        #matrix_concentration = concentrations[0]
        #solute_concentrations = concentrations[1:]
        #res = np.array([x*random.random() for x in solute_concentrations])
        #norm_res = (res/res.sum())*(1-matrix_concentration)
        #norm_conc = np.array([matrix_concentration] + norm_res.tolist())

        res = np.array([x*random.random() for x in concentrations])
        norm_conc = (res/res.sum())

        average_lattice = get_averaged_lattice(lattice_sizes, norm_conc)
        extras = {
            'matrix_elements':matrix_elements,
            'lattice_sizes':lattice_sizes,
            'average_lattice':average_lattice,
            'concentrations': norm_conc,
            'supercell_shape':supercell_shape,
            'random_displacement_stdev':random_displacement
                      }
        base_structure = gen_ase_supercell(average_lattice, supercell_shape, matrix_elements[0])

        random_structure = copy.deepcopy(base_structure)
        matrix_seed = random.randint(1, 2**32-1)
        extras['matrix_seed'] = matrix_seed
        random_ase = randomize_asestructure(random_structure, matrix_elements,
                                            norm_conc, matrix_seed)
        displacement_seed = random.randint(1, 2**32-1)
        extras['displacement_seed'] = displacement_seed
        random_ase.rattle(stdev=random_displacement, seed=displacement_seed)

        if nostore:
            random_ase = ase.build.sort(random_ase)
            dumpfile = os.path.join(dumpdir, "POSCAR_"+str(i))
            random_ase.write(dumpfile, format="vasp")
        else:
            store_asestructure(random_structure, extras, structure_group, dryrun)

if __name__ == "__main__":
   launch()
