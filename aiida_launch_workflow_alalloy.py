#!/usr/bin/env python
import click
import time
import copy
import sys
from pprint import pprint

import aiida
aiida.load_profile()
from aiida.engine import workfunction
from aiida.orm import Dict


def retrieve_alluncalculated_structures(structure_group_label,
                                        workchain_group_label=None):
    from aiida.orm import Group
    from aiida.orm import StructureData
    from aiida.orm import WorkChainNode 
    from aiida.orm import QueryBuilder

    sqb = QueryBuilder()
    sqb.append(Group, filters={'label': structure_group_label}, tag='g')
    sqb.append(StructureData, project='id', tag='s', with_group='g')
    sqb.append(WorkChainNode, tag='job', with_incoming='s')

    filters = {}
    if workchain_group_label:
        filters = {'label': workchain_group_label}
    sqb.append(Group, with_node='job', filters=filters)

    ids_dealt_with = [_ for _, in sqb.distinct().all()] or [-1]  # prevent empty list

    # # Now the main query:
    qb = QueryBuilder()
    qb.append(Group, filters={'label': structure_group_label}, tag='g')
    qb.append(StructureData, project='*', tag='s', with_group='g',
              filters={'id': {'!in': ids_dealt_with}})  # filter out calculated '!in' for not in

    res = [x[0] for x in qb.all()]
    return res

def retrieve_numactive_calculations():
    from aiida.orm import QueryBuilder
    from aiida.orm import CalcJobNode
    qb = QueryBuilder()
    qb.append(CalcJobNode,
              filters={'attributes.process_state':
                       {'!in': ['finished', 'excepted', 'killed']}}
    )
    return len(qb.all())

def retrieve_numactive_elastic():
    from aiida.orm import QueryBuilder
    from aiida.orm import CalcJobNode
    qb = QueryBuilder()
    qb.append(CalcJobNode,
              filters={'attributes.process_state':
                       {'!in': ['finished', 'excepted', 'killed']},
                       'attributes._process_label':'ElasticWorkChain'}
    )
    return len(qb.all())


def get_numelectrons_structure_upffamily(structure, pseudos):

    def parse_numelectrons_upfdata(upfdata):
        upfcontent = upfdata.get_content().split('\n')
        for line in upfcontent:
            if "valence" in line.lower() and "z" in line.lower():
                if len(line.split("=")) == 2:
                    num_e = int(float((line.split("=")[-1].strip().strip('"'))))
                elif len(line.split()) == 3:
                    num_e = int(float(line.split()[0]))
                else:
                    raise Exception("Could not parse {}".format(upfdata))
        return num_e

    def build_upf_numelectrons_dict(structure_ase, pseudos):
        element_nume_dict = {}
        for element in set(structure_ase.get_chemical_symbols()):
            print(pseudos[element])
            upfdata = pseudos[element]
            element_nume_dict[element] = parse_numelectrons_upfdata(upfdata)
        return element_nume_dict

    structure_ase = structure.get_ase()

    element_nume_dict = build_upf_numelectrons_dict(structure_ase, pseudos)

    num_e = 0
    for element in structure_ase.get_chemical_symbols():
        num_e += element_nume_dict[element]

    return num_e


def get_kmeshfrom_kptper_recipang(aiida_structure, kptper_recipang):
    import numpy as np

    ase_structure = aiida_structure.get_ase()
    reci_cell = ase_structure.get_reciprocal_cell()
    kmesh = [np.ceil(kptper_recipang * np.linalg.norm(reci_cell[i]))
             for i in range(len(reci_cell))]
    return kmesh


def get_nummachines(structure, pseudo_familyname):
    # NOTE: used very adhoc guess for nodes, assuming quadratic scaling
    from aiida.orm.nodes.data.upf import get_pseudos_from_structure
    pseudos = get_pseudos_from_structure(structure, pseudo_familyname)
    num_electrons = get_numelectrons_structure_upffamily(structure, pseudos)
    a2 = 1.5*10**-6
    a1 = 5.7*10**-3
    a0 = 2
    numnodes = a2*num_electrons**2+a1*num_electrons+a0
    numnodes = max(round(numnodes/2)*2, 2)  # force even # of nodes
    return numnodes


def get_nk(num_machines, code):
    def nk_nump_evenlydivisible(nk, nump):
        nk = float(nk)
        nump = float(nump)

        if round(nump/nk) == nump/nk:
            return True
        else:
            return False

    nk = str(int(num_machines/2))  # adhoc guess

    print("WARNING: Skipping manual check for nk divisibility!")
    ## check if our choice is valid
    #computer = code.computer
    #ppm = computer.get_default_mpiprocs_per_machine()
    #if ppm is None:
    #    ppm = 1
    ## if a local computer we set nk = 1
    #if ppm == 1:
    #   nk = str(1)
    #   return nk
    #nump = num_machines * ppm
    #if not nk_nump_evenlydivisible(nk, nump):
    #    raise Exception("Error number processors: {} "
    #                    "is not divisible by nk: {}".format(nump, nk))

    print("For Nodes={} using nk={}".format(num_machines,nk))
    return nk

def wf_getconventionalstructure(structuredata):
    '''
    Standardize an AiiDA StructureData object via pymatgen Structure
        using spglib
    :param structuredata: original StructureData
    '''
    raise Exception("THIS SHOULD BE A WORKFUNCTION!") 
    from aiida.orm import StructureData
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    mg_structure = structuredata.get_pymatgen()
    sga = SpacegroupAnalyzer(mg_structure)
    standard_structure = sga.get_conventional_standard_structure()
    standard_structuredata = StructureData(pymatgen_structure=standard_structure)

    return standard_structuredata


def wf_getkpoints(aiida_structure, kptper_recipang):
    from aiida.orm import KpointsData

    def get_kmeshfrom_kptper_recipang(aiida_structure, kptper_recipang):
        import numpy as np
        kptper_recipang = kptper_recipang.value

        ase_structure = aiida_structure.get_ase()
        reci_cell = ase_structure.get_reciprocal_cell()
        kmesh = [np.ceil(kptper_recipang * np.linalg.norm(reci_cell[i]))
                 for i in range(len(reci_cell))]
        return kmesh

    kpoints_mesh = get_kmeshfrom_kptper_recipang(aiida_structure, kptper_recipang)
    kpoints = KpointsData()
    kpoints.set_kpoints_mesh(kpoints_mesh)

    return kpoints

def wf_setupparams(base_parameter, structure,
                   pseudo_familyname, nume2bnd_ratio,
                   with_simple_spin,
                   cellpress_parameter,
                   relax_method=None):
        from aiida.orm.nodes.data.upf import get_pseudos_from_structure

        import collections
        def update(d, u):
            for k, v in u.items():
                if isinstance(v, collections.Mapping):
                    d[k] = update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        pseudos = get_pseudos_from_structure(structure, pseudo_familyname.value)
        nelec = get_numelectrons_structure_upffamily(structure, pseudos)
        nbnd = nelec * nume2bnd_ratio.value
        nbnd = max(nbnd, 20) # minimum of 20 bands to avoid certain crashes

        ## for some reason La needs more bands than other elements 
        #  (I thought this was the case.. but maybe smearing too high!)
        #nbnd_2x_elements = ['La']
        #symbols = structure.get_ase().symbols
        #if any([x in symbols for x in nbnd_2x_elements]):
        #    nbnd = int(nbnd*2)

        parameter_dict = base_parameter.get_dict()
        parameter_dict['SYSTEM']['nbnd'] = nbnd
        if relax_method is not None:
            parameter_dict['CONTROL']['calculation'] = relax_method
        if with_simple_spin:
            # set nspin
            parameter_dict['SYSTEM']['nspin'] = 2 

            # set initial magnetism
            structure_ase = structure.get_ase()
            mag_dict = {x:0.5 for x in set(structure_ase.get_chemical_symbols())}
            parameter_dict['SYSTEM']['starting_magnetization'] = mag_dict 
            


        cellpress_dict = cellpress_parameter.get_dict()
        parameter_dict.update(cellpress_dict)
        parameters = Dict(dict=parameter_dict)


        return parameters

def wf_delete_vccards(parameter):
    new_dict = parameter.get_dict()
    if 'CELL' in new_dict:
        del new_dict['CELL']
    return Dict(dict=new_dict)



@click.command()
@click.option('-c', '--code_node', required=True,
              help="node of code to use")
@click.option('-sg', '--structure_group_label', required=True,
              help='input group of structures to submit workchains on')
@click.option('-wg', '--workchain_group_label', required=True,
              help='output group of workchains')
@click.option('-sn', '--structure_node',
              help='structure node to submit workchains on.'
              'creates the provided structure group and adds the node')
@click.option('-bp', '--base_parameter_node', required=True,
              help='node of base ParameterData to setup calculations')
@click.option('-pfn', '--pseudo_familyname', required=True,
              help='name of pseudopotential family to use')
@click.option('-kra', '--kptper_recipang', required=True,
              help='number of kpoints to use per reciprocal angstrom')
@click.option('-ber', '--nume2bnd_ratio', required=True,
              help='band to electron ratio')
@click.option('-cm', '--calc_method', default='scf',
              type=click.Choice(["scf", "relax", "vc-relax", "elastic"]),
              help='The type of calculation to perform')
@click.option('-wsp', '--with_simple_spin', is_flag=True,
              help='Runs with nspin=2 and sets all starting magnetism to 0.5')
@click.option('-ucs', '--use_conventional_structure', is_flag=True,
              help='Turns the input structure to its pymatgen conventional form prior to running')
@click.option('-pct', '--press_conv_thr', default=None,
              help='Specify the pressure conv threshold in Kbar (vc-relax only)')
@click.option('-ja', '--job_account', default=None,
              help='account to use when submitting job')
@click.option('-mws', '--max_wallclock_seconds', default=8*60*60,
              help='maximum wallclock time per job in seconds')
@click.option('-mac', '--max_active_calculations', default=300,
              help='maximum number of active calculations')
@click.option('-mae', '--max_active_elastic', default=5,
              help='maximum number of active elastic workchains')
@click.option('-mns', '--max_nodes_submit', default=20,
              help='maximum nodes that can be used in a submission')
@click.option('-mas', '--max_atoms_submit', default=400,
              help='maximum number atoms that can be used in a submission')
@click.option('-nnd', '--number_of_nodes', default=None,
              help='Force all calculations to use the specified number of nodes')
@click.option('-memgb', '--memory_gb', default=None,
              help='specify the amount of memory for all jobs in GB')
@click.option('-nd', '--ndiag', default=None,
              help='ndiag setting to be passed direct to QE')
@click.option('-nk', '--npools', default=None,
              help='npools setting to be passed direct to QE')
@click.option('-sli', '--sleep_interval', default=10*60,
              help='time to wait (sleep) between calculation submissions')
@click.option('-zmo', '--z_movement_only', is_flag=True,
              help='Restricts movement to the z direction only. For relaxing stacking fault')
@click.option('-zco', '--z_cellrelax_only', is_flag=True,
              help='Restricts vc-relax to the z direction only. For relaxing stacking fault')
@click.option('-stm', '--strain_magnitudes', default=None,
              help='A comma seperated list of strain magnitudes. Only used for elastic workchain')
@click.option('-uas', '--use_all_strains', is_flag=True,
              help='Force use of all strains. Only used for elastic workchain')
@click.option('-kwd', '--keep_workdir', is_flag=True,
              help='Keep the workdir files after running')
@click.option('-dr', '--dryrun', is_flag=True,
              help="Prints inputs but does not launch anything")
@click.option('-sdb', '--submit_debug', is_flag=True,
              help='submit the script to debug queue. Submits one structure only'
                   ' and does not attach the output to the workchain_group')
@click.option('-rdb', '--run_debug', is_flag=True,
              help='run the script in debug mode. runs first calc then exits'
                   ' and does not attach the output to the workchain_group')
def launch(code_node, structure_group_label, workchain_group_label,
           structure_node, base_parameter_node,
           pseudo_familyname, kptper_recipang,
           nume2bnd_ratio, press_conv_thr,
           calc_method, with_simple_spin, use_conventional_structure,
           job_account,
           max_wallclock_seconds, max_active_calculations, max_active_elastic,
           max_nodes_submit, max_atoms_submit,
           number_of_nodes, memory_gb, ndiag, npools,
           sleep_interval, z_movement_only, z_cellrelax_only,
           strain_magnitudes, use_all_strains,
           keep_workdir, dryrun, submit_debug, run_debug):
    from aiida.orm import Group, load_node
    from aiida.orm import Bool, Dict, Float, List, Int, Str, StructureData
    from aiida.engine import submit, run
    from aiida.plugins.factories import WorkflowFactory
    # announce if running in debug mode
    if submit_debug:
        print("Running in debug mode!")

    # setup parameters
    code = load_node(code_node)
    workchain_group = Group.objects.get_or_create(label=workchain_group_label)[0]
    base_parameter = load_node(base_parameter_node)

    if structure_node:
        structure_group = Group.objects.get_or_create(label=structure_group_label)[0]
        input_structure = load_node(structure_node)
        if not isinstance(input_structure, StructureData):
            raise Exception("structure node was not a StructureData")
        structure_group.add_nodes([input_structure])

    # Load all the structures in the structure group, not-yet run in workchain_group_label
    structure_group = Group.get(label=structure_group_label)
    uncalculated_structures = retrieve_alluncalculated_structures(
                                structure_group_label,
                                workchain_group_label=workchain_group_label
    )

    if len(uncalculated_structures) == 0:
        print(("All structures in {} already have associated workchains in "
              "the group {}".format(structure_group_label, workchain_group_label)))
        sys.exit()

    # determine number of calculations to submit
    running_calculations = retrieve_numactive_calculations()
    calcs_to_submit = max_active_calculations - running_calculations
    if calc_method == 'elastic':
        running_elastic = retrieve_numactive_elastic()
        calcs_to_submit = max_active_elastic - running_elastic


    # submit calculations
    submit_counter=0
    for structure in uncalculated_structures:
        if use_conventional_structure:
            structure = wf_getconventionalstructure(structure)
        print("Preparing to launch {}".format(structure))
        print("calcs to submit: {} (active/max){}:{}".format(
                                     len(uncalculated_structures) -submit_counter,
                                                        calcs_to_submit,
                                                        max_active_calculations))
        submit_counter += 1

        if len(structure.get_ase()) > max_atoms_submit:
            print("{} has more atoms than the max allowed {}".format(structure,
                                                                     max_atoms_submit))
            print("If you wish to overide please use --max_atoms_submit")
            continue

        # ensure no more than the max number of calcs are submitted
        while (calcs_to_submit <= 0):
            running_calculations = retrieve_numactive_calculations()
            calcs_to_submit = max_active_calculations - running_calculations
            if calc_method == 'elastic':
                running_elastic = retrieve_numactive_elastic()
                calcs_to_submit = max_active_elastic - running_elastic
            if calcs_to_submit <= 0:  # in case jobs finished during submission
                if calc_method == 'elastic':
                    print(("{} elastic running,"
                          "max num elastic {} waiting....".format(
                              running_elastic, max_active_elastic)))
                else:
                    print(("{} calcs running,"
                          "max num calcs {} waiting....".format(
                              running_calculations, max_active_calculations)))
                time.sleep(sleep_interval)

        # start timer to inspect job submission times
        from timeit import default_timer as timer
        start = timer()

        # add any cell-related parameters specified from cli
        if "CELL" in base_parameter.get_dict():
            cellpress_dict = {"CELL":base_parameter.get_dict()["CELL"]}
        else:
            cellpress_dict = {}
            if press_conv_thr or z_cellrelax_only:
                    cellpress_dict["CELL"] = {}
        if press_conv_thr:
            cellpress_dict["CELL"]["press_conv_thr"] = float(press_conv_thr)
        if z_cellrelax_only:
            cellpress_dict["CELL"]["cell_dofree"] = "z"

        # determine number of bands & setup the parameters
        cellpress_parameter = Dict(dict=cellpress_dict)
        parameters = wf_setupparams(base_parameter,
                                    structure,
                                    Str(pseudo_familyname),
                                    Float(nume2bnd_ratio),
                                    Bool(with_simple_spin),
                                    cellpress_parameter,
                                    relax_method=calc_method)

        # determine kpoint mesh & setup kpoints
        kpoints = wf_getkpoints(structure, Int(kptper_recipang))

        # determine parallelization & resources (setup the settings & options)
        if number_of_nodes:
            num_machines = int(number_of_nodes)
        else:
            num_machines = get_nummachines(structure, pseudo_familyname)
            if calc_method in ['relax', 'vc-relax']:
               #num_machines += 4
               print("WARNING: Will not automatically increase number of nodes for relax job!")
            if num_machines > int(max_nodes_submit):
                print("{} nodes requested, maximum is {}".format(num_machines, max_nodes_submit))
                print("If you wish to launch please choose nodes manually with --number_of_nodes")
                continue
        print("WARNING: JUST GUESSING NUM_MPIPROCS_PER_MACHINE")
        options_dict = {
            'max_wallclock_seconds': max_wallclock_seconds,
            'resources': {'num_machines': num_machines,
                          'num_mpiprocs_per_machine': 32},
        }
        if job_account is not None:
            print("WARNING: overriding custom_scheduler_command")
            account_cmd = "#SBATCH --account={}".format(job_account)
            options_dict['custom_scheduler_commands'] = account_cmd
        if memory_gb:
            options_dict['max_memory_kb'] = int(int(memory_gb)*1024*1024)
        if submit_debug:
            num_machines = 2
            options_dict['resources']['num_machines'] = num_machines
            options_dict['max_wallclock_seconds'] = int(30*60)
            options_dict['queue_name'] = 'debug'
        workchain_options = options_dict

        if npools is not None:
            nk = npools
            print("nk manually set to {}".format(npools))
        else:
            nk = get_nk(num_machines, code)
            print("nk determined to be {}".format(nk))
        settings_dict = {
            'cmdline': ['-nk', nk],
            'no_bands': True
            }
        if ndiag:
            settings_dict['cmdline'] += ['-ndiag', ndiag]
        if z_movement_only:
            num_atoms = len(structure.get_ase())
            coordinate_fix = [[True,True,False]]*num_atoms
            settings_dict['fixed_coords'] = coordinate_fix
        settings = Dict(dict=settings_dict)

        # need to set up pseudos earlier 
        from aiida.orm.nodes.data.upf import get_pseudos_from_structure
        pseudos = get_pseudos_from_structure(structure, pseudo_familyname)

        # setup inputs & submit workchain
        clean_workdir = not keep_workdir
        inputs = {
                  'clean_workdir': Bool(clean_workdir),
                  }
        base_inputs = {
            'pw': {
                'code': code,
                'parameters': wf_delete_vccards(parameters),
                'metadata': {'options': workchain_options,
                             },
                'settings': settings,
                'pseudos': pseudos,
                'overrides': Dict({'delta_factor_mixing_beta':1.0})
            }
        }
        relax_inputs = {
            'base': {k: base_inputs[k]  for k in base_inputs if k != 'parameters'},
            #'relaxation_scheme': Str('relax'),
            #'final_scf' : Bool(False),
            #'meta_convergence' : Bool(False)
        }
        if calc_method == 'scf':
            WorkChain = WorkflowFactory('quantumespresso.pw.base')
            inputs.update(base_inputs)
            inputs['pw']['structure'] = structure
            inputs['kpoints'] = kpoints
            #inputs['pseudo_family'] = Str(pseudo_familyname)
        elif calc_method in ['relax', 'vc-relax']:
            WorkChain = WorkflowFactory('quantumespresso.pw.relax')
            inputs.update(relax_inputs)
            inputs['structure'] = structure
            inputs['base']['pseudo_family'] = Str(pseudo_familyname)
            inputs['base']['kpoints'] = kpoints
        elif calc_method == 'elastic':
            raise Exception("Elastic mode calculations no longer supported!")
            WorkChain = WorkflowFactory('elastic')
            inputs['structure'] = structure

            # Unfortunately deepcopy on code caueses issues so we need to duplicate
            # a lot of information
            sub_relax_inputs = {
                'base': {k: base_inputs[k]  for k in base_inputs if k != 'parameters'},
                'relaxation_scheme': Str('relax'),
                'final_scf' : Bool(False),
                'meta_convergence' : Bool(False)
            }
            sub_relax_inputs['base']['pseudo_family'] = Str(pseudo_familyname)
            sub_relax_inputs['base']['kpoints'] = kpoints
            sub_relax_inputs['relaxation_scheme'] = Str('relax')
            sub_relax_parameters = wf_delete_vccards(parameters)
            sub_relax_inputs['base']['pw']['parameters'] = sub_relax_parameters


            sub_vcrelax_inputs = {
                'base': {k: base_inputs[k]  for k in base_inputs if k != 'parameters'},
                'relaxation_scheme': Str('relax'),
                'final_scf' : Bool(False),
                'meta_convergence' : Bool(False)
            }
            sub_vcrelax_inputs['base']['pseudo_family'] = Str(pseudo_familyname)
            sub_vcrelax_inputs['base']['kpoints'] = kpoints
            sub_vcrelax_inputs['relaxation_scheme'] = Str('relax')
            sub_vcrelax_inputs['relaxation_scheme'] = Str('vc-relax')
            sub_vcrelax_inputs['base']['pw']['parameters'] = parameters

            inputs['initial_relax'] = sub_vcrelax_inputs
            inputs['elastic_relax'] = sub_relax_inputs

            if strain_magnitudes:
                strain_magnitudes_list = [float(x) for x in strain_magnitudes.split(',')]
                inputs['strain_magnitudes'] = List(list=strain_magnitudes_list)
            if use_all_strains:
                inputs['symmetric_strains_only'] = Bool(False)
        else:
            raise Exception("Invalid calc_method: {}".format(calc_method))

        def print_timing(start):
            end = timer()
            time_elapsed = end - start
            print("timing: {}s".format(time_elapsed))

        calcs_to_submit -= 1
        if dryrun:
            pprint("ase_structure: {}".format(structure.get_ase()))
            pprint("aiida_settings: {}".format(settings.get_dict()))
            #pprint "aiida_parameters: {}".format(inputs['base']['parameters'].get_dict())
            pprint("aiida_options: {}".format(workchain_options))
            pprint("aiida_inputs: ")
            pprint(inputs)
            print_timing(start)
            continue
        elif run_debug:
            run(WorkChain, **inputs)
            sys.exit()
        else:
            node = submit(WorkChain, **inputs)
            print("WorkChain: {} submitted".format(node))
            print_timing(start)

        if submit_debug:
            sys.exit()

        workchain_group.add_nodes([node])



if __name__ == "__main__":
    launch()

