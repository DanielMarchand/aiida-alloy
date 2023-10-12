#!/usr/bin/env python
import aiida
aiida.load_profile()
from qe_tools import CONSTANTS as constants
from aiida_create_solutesupercell_structures import *
from aiida.orm import load_node, Node, Group, QueryBuilder
from aiida.plugins.factories import WorkflowFactory
from aiida.orm import CalcJobNode, WorkChainNode
from aiida.orm import StructureData, TrajectoryData
import click
from ase import Atoms
from ase.io import write as ase_write
import aiida_utils
import sys
import os
import numpy as np

#Define unit conversions
ANGSTROM_TO_BOHRRADIUS = 1./constants.bohr_to_ang
EV_PER_ANGSTROM_TO_HARTREE_PER_BOHRRADIUS = constants.bohr_to_ang/constants.hartree_to_ev
EV_PER_ANGSTROM_P3_TO_HARTREE_PER_BOHRRADIUS_P3 = (constants.bohr_to_ang**3)/constants.hartree_to_ev
GPA_TO_EV_PER_ANGSTROM_P3 = 1./160.21766208
GPA_TO_HARTREE_PER_BOHRADIUS_P3 = GPA_TO_EV_PER_ANGSTROM_P3 * EV_PER_ANGSTROM_P3_TO_HARTREE_PER_BOHRRADIUS_P3
EV_TO_HARTREE = 1/constants.hartree_to_ev

def get_allnodes_fromgroup(group_label):
    qb = QueryBuilder()
    qb.append(Group, filters={'label': group_label}, tag='g')
    qb.append(Node, tag='job', with_group='g')
    all_nodes = [x[0] for x in qb.all()]
    return all_nodes

def get_outputcalcs(node):
    q = QueryBuilder()
    q.append(WorkChainNode, filters={"uuid": node.uuid}, tag="worknode")
    q.append(WorkChainNode, tag="worknode2",
             with_incoming="worknode", project=["id", "ctime",  "*"])
    q.order_by({"worknode2": "ctime"})
    child_nodes = [x[2] for x in q.all()]
    return child_nodes


def write_runner_commentline(fileout, uuid, extra_comments={}):
    fileout.write("begin\ncomment ")
    fileout.write("uuid: {} ".format(uuid))
    for label in extra_comments:
        fileout.write("{}: {} ".format(label, extra_comments[label]))
    fileout.write("\n")
    return

def write_runner_cell(fileout, cell):
    cell = cell * ANGSTROM_TO_BOHRRADIUS
    for idx_cell, i in enumerate(cell):
        fileout.write("lattice %.10f %.10f %.10f\n" %
                      (cell[idx_cell][0], cell[idx_cell][1], cell[idx_cell][2]))
    return

def write_runner_stress(fileout, stress):
    stress = np.array(stress) * GPA_TO_HARTREE_PER_BOHRADIUS_P3
    for idx_stress, i in enumerate(stress):
        fileout.write("stress %.20f %.20f %.20f\n" %
                      (stress[idx_stress][0],
                       stress[idx_stress][1],
                       stress[idx_stress][2]))
    return

def write_runner_atomlines(fileout, atomiccoord_array, elements, atomicforce_array=None):
    """
    Assumes input units of eV and angstrom
    """
    if atomicforce_array is None:
        atomicforce_array = np.zeros(atomiccoord_array.shape)

    atomiccoord_array = atomiccoord_array * ANGSTROM_TO_BOHRRADIUS
    atomicforce_array = atomicforce_array * EV_PER_ANGSTROM_TO_HARTREE_PER_BOHRRADIUS

    for i in range(len(atomiccoord_array)):
        atomiccoord = atomiccoord_array[i]
        element = elements[i]
        atomicforce = atomicforce_array[i]

        fileout.write("atom   %.6f    %.6f   %.6f "
                      "%s  0.0   0.0  "
                      "%.10f  %.10f  %.10f\n" %
                      (atomiccoord[0], atomiccoord[1], atomiccoord[2],
                       element,
                       atomicforce[0], atomicforce[1], atomicforce[2]))
    return

def write_runner_finalline(fileout, energy=0, charge=0):
    fileout.write("energy %.15f\n" % (energy*EV_TO_HARTREE))
    fileout.write("charge %.15f\nend\n" % charge)
    return


def write_structure_torunner(fileout, structure_node, extra_comments={}):
    # get structure path, if applicable
    ase_structure = structure_node.get_ase()
    ase_structure.wrap() # need this to avoid a bug in old n2p2 versions

    cell = ase_structure.get_cell()
    positions = ase_structure.get_positions()
    elements = ase_structure.get_chemical_symbols()

    write_runner_commentline(fileout, structure_node.uuid, extra_comments=extra_comments)
    write_runner_cell(fileout, cell)
    write_runner_atomlines(fileout, positions, elements)
    write_runner_finalline(fileout)
    return

def get_timesorted_calcjobs(relaxworknode):
    q = QueryBuilder()
    q.append(WorkChainNode, filters={"uuid": relaxworknode.uuid}, tag="relaxworknode")
    q.append(CalcJobNode, with_incoming="relaxworknode",
             project=["id", "ctime",  "*"],  tag="calc")
    q.order_by({"calc": "ctime"})
    timesorted_scf = [x[2] for x in q.all()]
    return timesorted_scf

def get_timesorted_basenodes(relaxworknode):
    q = QueryBuilder()
    q.append(WorkChainNode, filters={"uuid": relaxworknode.uuid}, tag="relaxworknode")
    q.append(WorkChainNode, with_incoming="relaxworknode",
             project=["id", "ctime",  "*"],  tag="calc")
    q.order_by({"calc": "ctime"})
    timesorted_scf = [x[2] for x in q.all()]
    return timesorted_scf

def get_timesorted_scfs(worknode, relax_worknode=False):
    q = QueryBuilder()
    q.append(WorkChainNode, filters={"uuid": worknode.uuid}, tag="worknode")
    output_tag = "worknode"
    if relax_worknode:
        output_tag = "worknode2"
        q.append(WorkChainNode, tag=output_tag, with_incoming="worknode")
    q.append( CalcJobNode,
              with_incoming=output_tag, project=["id", "ctime",  "*"],  tag="calc")
    q.order_by({"calc": "ctime"})
    timesorted_scf = [x[2] for x in q.all()]
    return timesorted_scf

def write_pwbase_torunner(fileout, pwbasenode, dump_stress, extra_comments={}):
    scf_node = get_timesorted_scfs(pwbasenode)[-1]

    try:
        ase_structure = scf_node.inputs.structure.get_ase()
        ase_structure.wrap()
    except Exception:
        ase_structure = scf_node.inputs.pw__structure.get_ase()
        ase_structure = ase_structure.wrap()
    cell = ase_structure.get_cell()
    positions = ase_structure.get_positions()
    elements = ase_structure.get_chemical_symbols()

    try:
        atomicforce_array = scf_node.outputs.output_array.get_array('forces')[-1]
    except Exception:
        atomicforce_array = scf_node.outputs.output_trajectory.get_array('forces')[-1]

    if dump_stress:
        #NOTE: in modern runs this data is in the output_trajectory
        stress = scf_node.outputs.output_parameters.attributes['stress']
    energy = scf_node.outputs.output_parameters.attributes['energy']

    write_runner_commentline(fileout, pwbasenode.uuid, extra_comments=extra_comments)
    write_runner_cell(fileout, cell)
    if dump_stress:
        write_runner_stress(fileout, stress)
    write_runner_atomlines(fileout, positions, elements, atomicforce_array=atomicforce_array)
    write_runner_finalline(fileout, energy=energy)
    return

def get_relaxnode_calcoutputs(node):
    q = QueryBuilder()
    q.append(WorkChainNode, filters={"uuid": node.uuid}, tag="worknode")
    q.append(WorkChainNode,
             tag="worknode2",
             with_incoming="worknode")
    q.append(CalcJobNode,
             tag="pwcalc",
             with_incoming="worknode2",
             project=["id", "ctime",  "*"])
    q.order_by({"pwcalc": "ctime"})
    child_nodes = [x[2] for x in q.all()]
    return child_nodes

def get_timesorted_values(relax_node, arraylabel, np_concatenate=True,
                          check_outputparams=False):
    # check_outputparams, some values are not added to trajectory if numsteps =1 (e.g energy)
    child_nodes = get_relaxnode_calcoutputs(relax_node)
    output_array = []
    num_steps = 0
    addextra_vcinfo = False # some vc-relax calcs omit final data in the trajectory
    for node in child_nodes:
        # legacy option to keep symbols retrieval the same
        if arraylabel == "symbols":
            return node.attributes['symbols']
        # vc-relax calcuations require some double-counting
        if node.inputs.parameters.get_dict()['CONTROL']['calculation'] == 'vc-relax':
            addextra_vcinfo = True
        try:
           exit_status  = node.exit_status
        except AttributeError:
           exit_status = 1
        if exit_status not in  [0, 2, 400, 500, 501, 520]:
        #if exit_status not in  [0, 400, 500, 501, 520]:
            print("Skipping failed child {} of {}".format(node, relax_node))
            continue

        try:
           step_array = node.outputs.output_trajectory.get_array('steps')
           energy_array = node.outputs.output_trajectory.get_array('energy')
           child_array = node.outputs.output_trajectory.get_array(arraylabel)
           num_steps = len(step_array)
        except Exception:
            print("No trjactories in child {} of {}".format(node, relax_node))
            continue

        # in new versions of QE, the final cell, positions is not contained in the 
        # trajectory but for training data it should be enough to snip the last step
        if len(step_array) == len(child_array) - 1:
            child_array = child_array[:-1]
            addextra_vcinfo = False # do not add a final step later 

        if len(child_array) != num_steps:
            print("ERROR {} does not match number of steps! For {} of {}".format(arraylabel, node, relax_node))
            continue

        if num_steps == 1 and check_outputparams:
            output_array.append([node.outputs.output_parameters.get_dict()[arraylabel]])
        else:
            output_array.append(child_array)

    # vc-relax nodes are really fussy when it comes to appending to trajectory data
    if relax_node.exit_status != 0 or num_steps == 1:
       addextra_vcinfo = False

    if addextra_vcinfo and arraylabel in ['steps', 'cells', 'positions']:
        output_array.append([output_array[-1][-1]])

    try:
        output_array = np.concatenate(output_array)
    except ValueError:
        return []
    return output_array

def write_pwrelax_torunner(fileout, relax_node, write_only_relaxed,
                           energy_tol, verbose,  extra_comments={}):
    timesorted_steps = get_timesorted_values(relax_node, 'steps')
    num_steps = len(timesorted_steps)
    if num_steps == 0:
        print("WARNING: {} is empty, skipping!".format(relax_node))
        return
    timesorted_cells = get_timesorted_values(relax_node, 'cells')
    timesorted_positions = get_timesorted_values(relax_node, 'positions')
    timesorted_energy = get_timesorted_values(relax_node, 'energy', check_outputparams=True)
    timesorted_forces = get_timesorted_values(relax_node, 'forces')
    # assuming the element order remains unchanged
    try:
        elements = relax_node.inputs.structure.get_ase().get_chemical_symbols()
    except Exception:
        elements = relax_node.inputs.pw__structure.get_ase().get_chemical_symbols()

    #Sometimes the final forces are not parsed. Trim out the last energy in that case
    if relax_node.exit_status == 401:
        # new versions of AiiDA don't seem to have this problem anymore
        if len(timesorted_energy) != len(timesorted_forces):
            timesorted_energy = timesorted_energy[0:len(timesorted_forces)]

    print(relax_node)
    if len(timesorted_cells) != num_steps:
       raise AssertionError("Cells does not have the proper number of steps!")
    if len(timesorted_positions) != num_steps:
       raise AssertionError("Positions does not have the proper number of steps!")
    if len(timesorted_energy) != num_steps:
       print('E', len(timesorted_energy), 'STEPS', num_steps)
       print("ERROR {}".format(relax_node))
       raise AssertionError("Energies does not have the proper number of steps!")
    if len(timesorted_forces) != num_steps:
       raise AssertionError("Forces does not have the proper number of steps!")

    extra_comments={"trajectory_step":None}

    if verbose:
       print('relaxation steps:',len(timesorted_steps))

    try:
        final_scf = bool(relax_node.inputs.final_scf)
    except Exception:
        final_scf = False 

    if not write_only_relaxed:
        trajectory_looprange = list(range(len(timesorted_cells)))
    elif not final_scf:
        trajectory_looprange = [-1]
    else:
        trajectory_looprange = []
    final_loop = trajectory_looprange[-1]
    old_energy = timesorted_energy[0]
    for i in trajectory_looprange:
        del_e = np.abs(timesorted_energy[i] - old_energy)
        if i != 0 and i != final_loop and del_e  < energy_tol:
            print("Skipping step {} E_new: {} E_old:{} delE: {}".format(
                 i, timesorted_energy[i], old_energy, del_e))
            continue
        else:
            old_energy = timesorted_energy[i]
        extra_comments["trajectory_step"] = i
        write_runner_commentline(fileout, relax_node.uuid, extra_comments=extra_comments)
        write_runner_cell(fileout, timesorted_cells[i])
        write_runner_atomlines(fileout,
           timesorted_positions[i],
           elements,
           atomicforce_array=timesorted_forces[i])
        write_runner_finalline(fileout, energy=timesorted_energy[i])

    if final_scf:
        print('final')
        final_basenode = get_timesorted_basenodes(relax_node)[-1]
        extra_comments["trajectory_step"] = "final_scf"
        extra_comments["parent_uuid"] = relax_node.uuid
        write_pwbase_torunner(fileout, final_basenode, extra_comments=extra_comments)

    return


# show default values in click
orig_init = click.core.Option.__init__
def new_init(self, *args, **kwargs):
    orig_init(self, *args, **kwargs)
    self.show_default = True
click.core.Option.__init__ = new_init
# get help also with -h
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-gn', '--group_label', default=None,
              type=str, help="Group to export identified by label")
@click.option('-f', '--filename', required=False, default=False,
         type=str, help="filename for outputfile, default = work_group+\".input.data\"")
@click.option('-wor', '--write_only_relaxed', required=False, default=False,
         is_flag=True, help="only write the final relaxed structure")
@click.option('-et', '--energy_tol', required=False, default=0.5,
         help="Only dumps relaxation steps of minimum energy_tol(eV) apart")
@click.option('-sreadme', '--supress_readme', is_flag=True,
         help="DOES NOTHING KEPT ONLY FOR COMPATIBILITY."
              "supresses the generation of a readme file")
@click.option('-v', '--verbose', is_flag=True,
         type=str, help="Enables verbosity")
@click.option('-oe', '--output_elements', required=False,
              help="only output structures containing these elements")
@click.option('-re', '--required_elements', required=False,
              help="All output structures must contain this element")
@click.option('-ds', '--dump_stress', is_flag=True,
         type=str, help="dumps the stress for each output")


def createjob(group_label, filename, write_only_relaxed, energy_tol,
              supress_readme, verbose, output_elements, required_elements, dump_stress):
    ''' e.g.
    ./aiida_export_group_to_runner.py -gn Al6xxxDB_structuregroup
    '''
    energy_tol = energy_tol*EV_TO_HARTREE
    print("ENERGY_TOL", energy_tol)
    all_nodes = get_allnodes_fromgroup(group_label)

    if output_elements:
        output_elements = prep_elementlist(output_elements)
    if required_elements:
        required_elements = prep_elementlist(required_elements)

    add_to_filename = "__all_steps"
    if write_only_relaxed == True:
        add_to_filename = "__only_relaxed"

    if filename == False:
        file = "aiida_exported_group_"+group_label+add_to_filename+".input.data"
        fileout = open(file, "w")
    else:
        file = filename
        fileout = open(filename, "w")

    if verbose:
        print('file           :', file)
        print('structure_group:', group_label)

    for node in all_nodes:
        exit_status = node.exit_status
        if exit_status is None:
            print("WARNING {} has an unkown exit status, skipping!".format(node))
            continue
        if int(exit_status) in [0]:
            pass #succesful calculation no need to print a warning
        elif int(exit_status) in [401]:
            print("WARNING {} had a non-critical non-zero exit!".format(node, exit_status))
        elif int(exit_status) in [104, 305]:
            print("WARNING {} had a critical non-zero exit, skipping!".format(node, exit_status))
            continue
        else:
            print("WARNING Unknown exit status", exit_status, " for node", node.pk)

        if verbose:
            print("Writing node: {}".format(node.uuid))
        if output_elements:
            try:
                input_ase = node.inputs.structure.get_ase()
            except Exception:
                input_ase = node.inputs.pw__structure.get_ase()
            only_output_elements = all([x in output_elements
                                        for x in input_ase.get_chemical_symbols()])
            if not only_output_elements:
                print("Skipping: {}/{}".format(input_ase, node))
                continue
        if required_elements:
            try:
                input_ase = node.inputs.structure.get_ase()
            except Exception:
                input_ase = node.inputs.pw__structure.get_ase()
            any_required_elements = any([x in required_elements
                                        for x in input_ase.get_chemical_symbols()])
            if not any_required_elements:
                print("Skipping: {}/{}".format(input_ase, node))
                continue

        if isinstance(node, StructureData):
            print('using write_structure_torunner')
            write_structure_torunner(fileout, node)
        elif isinstance(node, WorkChainNode):
            process_label = node.attributes['process_label']
            if process_label == "PwBaseWorkChain":
                print('using write_pwbase_torunner')
                try:
                    write_pwbase_torunner(fileout, node, dump_stress)
                except Exception:
                    print("ERROR WRITING SCFNODE: ",node.uuid)
            elif process_label == "PwRelaxWorkChain":
                print('using write_pwrelax_torunner')
                try:
                    write_pwrelax_torunner(fileout, node,write_only_relaxed,
                                           energy_tol,verbose)
                except Exception as e:
                    print("ERROR WRITING RELAXNODE: ",node.uuid)
                    print(e)
            elif process_label == "ElasticWorkChain":
                print("recursively adding Elastic nodes")
                elastic_children = get_outputcalcs(node)
                all_nodes += elastic_children
            else:
                print("Could not identify node, skipping")
        else:
            print("Could not identify node, skipping")

    fileout.close()
    return


if __name__ == "__main__":
    createjob()
