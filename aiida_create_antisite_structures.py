#!/usr/bin/env python
import aiida
aiida.load_profile()
from aiida.orm import Group
from aiida.orm import StructureData
from aiida_create_solutesupercell_structures import *
import ase
import ase.build
import click
import numpy as np
import random
from aiida.orm import QueryBuilder
from aiida.orm.utils import load_node

def get_allstructurenodes_fromgroup(structure_group):
    from aiida.orm import Group
    from aiida.orm import StructureData
    from aiida.orm import QueryBuilder

    if structure_group:
        structure_group_label = structure_group.label
    else:
        return []

    sqb = QueryBuilder()
    sqb.append(Group, filters={'label': structure_group_label}, tag='g')
    sqb.append(StructureData, with_group='g')

    return [x[0] for x in sqb.all()]

def get_smallestcellindex(ase_structure):
    smallest_cellnorm = np.linalg.norm(ase_structure.cell[0])
    smallest_index = 0
    for i in range(len(ase_structure.cell)):
        this_cellnorm = np.linalg.norm(ase_structure.cell[i])
        if this_cellnorm < smallest_cellnorm:
            smallest_index = i
            smallest_cellnorm = this_cellnorm
    return smallest_index

def generate_supercell(inputstructure, target_numatoms):
    output_cell = copy.deepcopy(inputstructure)
    repeats = [1,1,1]
    while True:
        if len(output_cell) >= target_numatoms:
            break
        else:
            smallest_index = get_smallestcellindex(output_cell)
            repeats[smallest_index] += 1
            output_cell = inputstructure.repeat(repeats)
    return repeats, output_cell

def get_unique_sites(structure_ase):
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    structure_mg = AseAtomsAdaptor.get_structure(structure_ase)
    sga = SpacegroupAnalyzer(structure_mg)
    symmstruct_mg = sga.get_symmetrized_structure()

    elements = [x[0].species_string for x in symmstruct_mg.equivalent_sites]
    count = [elements[:i+1].count(elements[i]) for i in range(len(elements))]

    site_indices = [x[0] for x in symmstruct_mg.equivalent_indices]
    elements_count = ["{}{}".format(x[0],x[1]) for x in zip(elements, count)]
    wyckoff = symmstruct_mg.wyckoff_symbols

    unique_sites = list(zip(site_indices, elements_count, wyckoff))
    return unique_sites



@click.command()
@click.option('-in', '--input_group',
              help='group containing structures to base antisites off of.')
@click.option('-is', '--input_structures',
              help='A comma-seprated list of nodes/uuid to import')
@click.option('-tss', '--target_supercellsize', default=None,
              help="Target size for supercell")
@click.option('-se', '--solute_elements', required=True,
              help="List of solute elements to create antsites with. "
              " Can pass a list of elements using comma seperation"
              " E.g. Mg,Si,Cu. Can specify the creation of a vacancy using 'Vac'")
@click.option('-sc', '--structure_comments', default="",
              help="Comment to be added to the extras")
@click.option('-sg', '--structure_group_label', required=True,
              help="Output AiiDA group to store created structures")
@click.option('-sgd', '--structure_group_description', default="",
              help="Description for output AiiDA group")
@click.option('-dr', '--dryrun', is_flag=True,
              help="Prints structures and extras but does not store anything")
def launch(input_group, input_structures,
           target_supercellsize,  solute_elements,
           structure_comments, structure_group_label,
           structure_group_description,
           dryrun):
    """
    Script for generating substitutional and vacancy defects for an input structure
    or all structures in an input group.
    """
    if not dryrun:
        structure_group = Group.objects.get_or_create(
                             label=structure_group_label,
                             description=structure_group_description)[0]
    else:
        structure_group = None

    if input_group:
        input_group = Group(input_group)
        structure_nodes = get_allstructurenodes_fromgroup(input_group)
    elif input_structures:
        input_structures = input_structures.split(',')
        structure_nodes = [load_node(x) for x in input_structures]
    else:
        raise Exception("Must use either input group or input structures")

    solute_elements = prep_elementlist(solute_elements)

    for structure_node in structure_nodes:
        extras = {
            'input_structure':structure_node.uuid,
            'structure_comments':structure_comments
                      }

        input_structure_ase = structure_node.get_ase()
        #Unfortunately, we must get the unique sites prior to supercell
        #creation, meaning changes in site index can cause bugs
        unique_sites = get_unique_sites(input_structure_ase)
        if target_supercellsize is not None:
            target_supercellsize = int(target_supercellsize)
            extras['target_supercellsize'] = target_supercellsize
            input_structure_ase = generate_supercell(
                                    input_structure_ase, target_supercellsize
                                    )[1]
        for unique_site in unique_sites:

            site_index, element_index, wyckoff = unique_site
            extras['site_index'] = site_index
            extras['element_index'] = element_index
            extras['wyckoff'] = wyckoff 

            for element in solute_elements:
                defect_structure = input_structure_ase.copy()
                extras['element_new'] = element
                if defect_structure[site_index].symbol == element:
                    continue
                else:
                   defect_structure[site_index].symbol = element
                   store_asestructure(defect_structure, extras,
                                      structure_group, dryrun)




if __name__ == "__main__":
   launch()
