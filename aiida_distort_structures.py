#!/usr/bin/env python
import aiida
aiida.try_load_dbenv()
from aiida.orm.group import Group
from aiida.orm.data.structure import StructureData
from aiida_create_solutesupercell_structures import *
import ase
import ase.build
import click
import numpy as np
import random
from aiida.orm.querybuilder import QueryBuilder
from aiida.orm.utils import load_node

def get_allstructures_fromgroup(group_name):
    qb = QueryBuilder()
    qb.append(Group, filters={'name': group_name}, tag='g')
    qb.append(StructureData, tag='job', member_of='g')
    all_nodes = [x[0] for x in qb.all()]
    return all_nodes


def randomize_asestructure(ase_structure, seed):
    random_f = np.random.RandomState(seed).rand()
    for i in range(len(ase_structure)):
        element_index = determine_selection(concentrations)
        element_toinsert = elements[element_index]
        if element_toinsert == "Vac":
            print "DELETED!!!"
            del ase_structure[i]
        else:
            ase_structure[i].symbol = element_toinsert
    return ase_structure

debug_global=0
def get_strained_structures(equilibrium_structure, strain_magnitudes,
                             symmetric_strains_only=True):
    import pymatgen as mg
    from pymatgen.analysis.elasticity import DeformedStructureSet
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.analysis.elasticity.tensors import symmetry_reduce


    global debug_global
    debug_global += 1 
    equilibrium_structure.write("/tmp/tmp_POSCAR/POSCAR_{}".format(debug_global),
                               format='vasp')
    try:
        equilibrium_structure_mg = AseAtomsAdaptor.get_structure(equilibrium_structure)
        deformed_mat_set = DeformedStructureSet(equilibrium_structure_mg,
                                            norm_strains=strain_magnitudes,
                                            shear_strains=strain_magnitudes)
    except Exception:
        equilibrium_structure.write("/tmp/POSCAR_fail", format='vasp')
        raise Exception("Somethign spoooky!")

    symmetry_operations_dict = {}
    deformations = deformed_mat_set.deformations
    if symmetric_strains_only:
        symmetry_operations_dict = symmetry_reduce(deformations,equilibrium_structure_mg)
        deformations = [x for x in symmetry_operations_dict]

    strained_structures = []
    for i in range(len(deformations)):
        mg_deformed_structure = deformations[i].apply_to_structure(equilibrium_structure_mg)
        deformed_structure = AseAtomsAdaptor.get_atoms(mg_deformed_structure)
        strained_structures.append(deformed_structure)
    deformations = [np.array(x).tolist() for x in deformations]

    return deformations, strained_structures

@click.command()
@click.option('-in', '--input_group',
              help='group containing structures to base randomization off of.')
@click.option('-is', '--input_structures',
              help='A comma-seprated list of nodes/uuid to import')
@click.option('-rs', '--repeat_expansion', default="1,1,1",
              help="A supercell expansion to apply prior to generating the structure")
@click.option('-vs', '--volumetric_strains', default="0",
              help="A comma sepearted list of volumetric strains to apply")
@click.option('-as', '--elastic_strains', default="0",
              help="A comma sepearted list of strain magnitudes to apply")
@click.option('-rdisp', '--random_displacement', default=0.15, type=float,
              help="stdev of random displacement in ang")
@click.option('-nrs', '--number_randomized_samples', default=0, type=int,
              help="Number of randomized samples to generate")
@click.option('-sc', '--structure_comments', default="",
              help="Comment to be added to the extras")
@click.option('-sg', '--structure_group_name', required=True,
              help="Output AiiDA group to store created structures")
@click.option('-sgd', '--structure_group_description', default="",
              help="Description for output AiiDA group")
@click.option('-dr', '--dryrun', is_flag=True,
              help="Prints structures and extras but does not store anything")
def launch(input_group, input_structures, repeat_expansion,
           volumetric_strains, elastic_strains, random_displacement,
           number_randomized_samples, structure_comments,
           structure_group_name, structure_group_description,
           dryrun):
    """
    Script for distoring the cell shape for an input structure
    """
    if not dryrun:
        structure_group = Group.get_or_create(
                             name=structure_group_name, description=structure_group_description)[0]
    else:
        structure_group = None

    if input_group:
        structure_nodes = get_allstructures_fromgroup(input_group)
    elif input_structures:
        input_structures = input_structures.split(',')
        structure_nodes = [load_node(x) for x in input_structures]
    else:
        raise Exception("Must use either input group or input structures")
    volumetric_strains = [float(x) for x in volumetric_strains.split(',')]
    elastic_strains = [float(x) for x in elastic_strains.split(',')]
    repeat_expansion = [int(x) for x in repeat_expansion.split(',')]

    for structure_node in structure_nodes:
        extras = {
            'input_structure':structure_node.uuid,
            'repeats':repeat_expansion,
            'structure_comments':structure_comments
                      }

        input_structure_ase = structure_node.get_ase()
        input_structure_ase = input_structure_ase.repeat(repeat_expansion)
        deformations, strained_structures = get_strained_structures(input_structure_ase,
                                                                    elastic_strains)
        for i in range(len(strained_structures)):
            extras['deformation'] = deformations[i]
            straindeformed_structure = copy.deepcopy(strained_structures[i])
            straindeformed_cell = copy.deepcopy(straindeformed_structure.cell)
            for j in range(len(volumetric_strains)):
                extras['random_seed'] = None
                extras['random_displacement_stdev'] = None
                extras['volume_strain'] = volumetric_strains[j]
                volume_deformation = 1.0+volumetric_strains[j]
                extras['volume_deformation'] = volume_deformation

                volumedeformed_structure = copy.deepcopy(straindeformed_structure)
                volumedeformed_structure.set_cell(straindeformed_cell*volume_deformation,
                                                  scale_atoms=True)
                store_asestructure(volumedeformed_structure, extras, structure_group, dryrun)
                for k in range(number_randomized_samples):
                    random_structure = copy.deepcopy(volumedeformed_structure)
                    random_seed = random.randint(1, 2**32-1)
                    extras['random_seed'] = random_seed
                    extras['random_displacement_stdev'] = random_displacement 
                    random_structure.rattle(stdev=random_displacement, seed=random_seed)
                    store_asestructure(random_structure, extras, structure_group, dryrun)

if __name__ == "__main__":
   launch()
