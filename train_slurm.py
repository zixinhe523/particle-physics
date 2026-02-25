'''
Author: Anthony Badea
Date: May 27, 2024
'''

# fix for keras v3.0 update
import os
# os.environ['TF_USE_LEGACY_KERAS'] = '1' # THIS NEEDS TO BE COMMENTED OUT IF ON NERSC

# python based
import tensorflow as tf
import tensorflow.keras.backend as K
import random
# from pathlib import Path
import time
import argparse
import json
import submitit
import h5py
import numpy as np
import shutil
import subprocess
import math
import itertools

# custom code
import dataloader

# omnifold
import omnifold

# print all imported package versions
def print_environment():

  print("TensorFlow version:", tf.__version__)

  # Get OmniFold version via importlib.metadata
  try:
      from importlib.metadata import version, PackageNotFoundError  # Python 3.8+
  except ImportError:
      from importlib_metadata import version, PackageNotFoundError  # backport

  try:
      omni_ver = version("omnifold")
  except PackageNotFoundError:
      omni_ver = "not found"

  # Optionally, show the module path
  omni_path = omnifold.__file__

  print("OmniFold version:", omni_ver)
  print("OmniFold path:", omni_path)

# SLURM sets CUDA_VISIBLE_DEVICES, so only the allocated GPU is visible to the task
# gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES')
# print(f"Assigned GPU: {gpu_id}")

# set gpu growth
def set_gpu_growth():
  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
    try:
      # Currently, memory growth needs to be the same across GPUs
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        try:
          import horovod.tensorflow as hvd
          hvd.init()
          tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
        except:
          print("No horovod")
      logical_gpus = tf.config.list_logical_devices('GPU')
      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      # Memory growth must be set before GPUs have been initialized
      print(e)
        
def train(
    conf
):

    # SLURM sets CUDA_VISIBLE_DEVICES, so only the allocated GPU is visible to the task
    set_gpu_growth()
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES')
    print(f"Assigned GPU: {gpu_id}")
    
    # print environment in all logs
    print_environment()

    # print conf
    print(conf)

    # print random seed information
    print("SLURM_JOBID:", os.environ.get("SLURM_JOBID"))
    print("SLURM_ARRAY_TASK_ID:", os.environ.get("SLURM_ARRAY_TASK_ID"))
    # Check Python's built-in RNG
    print("Python random sample:", random.random())
    # Check NumPy RNG
    print("NumPy random sample:", np.random.rand())
    # Check TensorFlow RNG
    print("TensorFlow random sample:", tf.random.uniform((1,)))
    
    # update %j with actual job number
    output_directory = conf["output_directory"]
    try:
        job_env = submitit.JobEnvironment()
        job_id = str(job_env.job_id)
    except:
        job_id = "%08x" % random.randrange(16**8)
        
    output_directory = os.path.abspath(output_directory.replace("%j", job_id))
    os.makedirs(output_directory, exist_ok=True)
    print(output_directory)
        
    # load the aleph reconstructed data, reconstructed mc, and generator mc
    reco_data, reco_mc, gen_mc, pass_reco, pass_gen = dataloader.DataLoader(conf)

    # run a closure test where data is replaced by reco MC
    if "run_closure_test" in conf.keys() and conf["run_closure_test"]:
       print("Running a closure test where data is replaced by reco MC")
       print(reco_data.shape)
       reco_data = reco_mc[pass_reco]
       print(reco_data.shape)

    # create the event weights
    weights_mc = np.ones(gen_mc.shape[0], dtype=np.float32)
    weights_data = np.ones(reco_data.shape[0], dtype=np.float32)

    if "theory_variation_weights_path" in conf.keys():
        print("Using theory variation weights")
        theory_variation_weights = np.load(conf["theory_variation_weights_path"])
        weights_mc = theory_variation_weights
        print(weights_mc.shape)
    
    # make omnifold dataloaders ready for training
    data = omnifold.DataLoader(
      reco = reco_data,
      weight = weights_data,
      normalize = True,
      bootstrap = True if conf["job_type"] == "BootstrapData" else False
    )
    
    mc = omnifold.DataLoader(
      reco = reco_mc,
      pass_reco = pass_reco,
      gen = gen_mc,
      pass_gen = pass_gen,
      weight = weights_mc,
      normalize = True,
      bootstrap = True if conf["job_type"] == "BootstrapMC" else False
    )

    # make weights directory
    weights_folder_id = "%08x" % random.randrange(16**8)
    weights_folder = os.path.abspath(os.path.join(output_directory, f"./model_weights_{weights_folder_id}"))
    os.makedirs(weights_folder, exist_ok=True)

    # save the starting weights
    outFileName = os.path.abspath(os.path.join(weights_folder, "starting_weights.npz"))
    np.savez(outFileName, weights_data=data.weight, weights_mc=mc.weight)
      
    # write conf to json in output directory for logging
    output_conf_name = os.path.abspath(os.path.join(weights_folder, "conf.json"))
    with open(output_conf_name, 'w') as file:
      json.dump(conf, file, indent=4)  # indent=4 for pretty printing
    
    # prepare networks
    ndim = reco_data.shape[1] # Number of features we are going to create = thrust
    model1 = omnifold.MLP(ndim, layer_sizes = conf["layer_sizes"], activation="relu")
    model2 = omnifold.MLP(ndim, layer_sizes = conf["layer_sizes"], activation="relu")

    print(model1.summary())
    print(model2.summary())

    # prepare multifold
    mfold = omnifold.MultiFold(
      name = 'mfold_job{}'.format(job_id),
      model_reco = model1,
      model_gen = model2,
      data = data,
      mc = mc,
      weights_folder = weights_folder,
      log_folder = output_directory,
      batch_size = conf["batch_size"],
      epochs = conf["epochs"],
      lr = conf["lr"],
      niter = conf["niter"],
      verbose = conf["verbose"],
      early_stop = conf["early_stop"],
    )
    
    # launch training
    mfold.Unfold()
    
    # get weights
    omnifold_weights = mfold.reweight(gen_mc, mfold.model2, batch_size=1000)
    np.save(os.path.abspath(os.path.join(weights_folder, "omnifold_weights.npy")), omnifold_weights)

    omnifold_weights_reco = mfold.reweight(reco_mc, mfold.model1, batch_size=1000)
    np.save(os.path.abspath(os.path.join(weights_folder, "omnifold_weights_reco.npy")), omnifold_weights_reco)

if __name__ == "__main__":

    # set up command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--slurm", help="path to json file containing slurm configuration", default=None)
    parser.add_argument("--njobs", help="number of jobs to actually launch. default is all", default=-1, type=int)
    parser.add_argument('--verbose', action='store_true', default=False, help='Run the scripts with more verbose output')
    parser.add_argument('--run_systematics', action='store_true', default=False, help='Run the track and event selection systematic variations')
    parser.add_argument('--run_bootstrap_mc', action='store_true', default=False, help='Run the bootstrapping for MC')
    parser.add_argument('--run_bootstrap_data', action='store_true', default=False, help='Run the bootstrapping for data')
    parser.add_argument('--run_ensembling', action='store_true', default=False, help='Run the ensembling by retraining without changing the inputs')
    parser.add_argument('--run_closure_test', action='store_true', default=False, help="Run a closure test where data is replaced by reco MC.")
    parser.add_argument('--run_hyperparameter_scan', action='store_true', default=False, help='Run the hyperparameter scan')
    parser.add_argument('--run_niter_scan', action='store_true', default=False, help='Run the number of iteration scan based on the optimized hyperparameters')
    parser.add_argument('--run_theory_uncert', action='store_true', default=False, help='Run the theory uncertainty scan')
    parser.add_argument('--top_dir', help="Top level directory for storing data. Default to nersc directory", default="/pscratch/sd/b/badea/aleph/unfold-ee-logtau/UniFold/results/")
    args = parser.parse_args()

    # create top level output directory
    top_dir = args.top_dir
    top_dir = os.path.abspath(os.path.join(top_dir, f'training-{"%08x" % random.randrange(16**8)}', "%j"))

    with open("training_conf.json") as f:
      training_conf = json.load(f)
    training_conf["output_directory"] = top_dir
    training_conf["verbose"] = args.verbose
    print(training_conf)

    # number of repeated trainings per omnifold configuration
    '''
    If we need N ensemble per omnifold. Then we need:
    40*N trainings for ensembeling uncertainty
    40*N trainings for data bootstrap uncertainty
    40*N trainings for mc bootstrap uncertainty
    18*N trainings for systematic variations
    = 138*N
    '''
    
    n_training_per_node = 4 # number of trainings per node or per job launched
    # n_ensemble_per_omnifold = n_training_per_node # this will be scaled by 4 from the running 4 copies on each node
    
    # list of configurations to launch
    confs = []

    # trainings per ensemble
    N_trainings_per_ensemble = 100
    
    # add configurations for track and event selection systematic variations
    # total_n_systematics = 10 # closest to 10 which divides by 4
    # n_systematics = math.ceil(total_n_systematics / n_training_per_node)
    n_systematics = N_trainings_per_ensemble
    if args.run_systematics:
      for i in range(n_systematics):

        # sysematic variations
        SystematicVariationList = ["ntpc7", "pt04", "ech10", "no_neutrals", "with_met"]
        NeutralParticleMCVariations = ["nes_up", "nes_down", "ner"] # reco MC variations to account for mis-modeling of detector response/efficiency for neutral particles
        SystematicVariationList += NeutralParticleMCVariations
                
        for SystematicVariation in SystematicVariationList:
          temp = training_conf.copy()            
          # only apply cut based variations to data
          if SystematicVariation not in NeutralParticleMCVariations:
            temp["data"] = temp["data"].replace("nominal", SystematicVariation)
          temp["reco"] = temp["reco"].replace("nominal", SystematicVariation)
          temp["job_type"] = "Systematics"
          temp["i_ensemble_per_omnifold"] = i
          confs.append(temp)

    # add configurations for theory uncertainty scan
    if args.run_theory_uncert:

      # # boost 2025 results
      # # theory_variation_dir = "/pscratch/sd/b/badea/aleph/unfold-ee-logtau/ReweightMC/results/training-200471c7/"
      # theory_variation_dir = "/home/badea/e+e-/aleph/UnfoldThrustResults/theory_reweighting/training-200471c7/"
      # theory_variations = [
      #   ["Pythia8", os.path.join(theory_variation_dir, "39912440_0/model_weights_b7634c53/Reweight_Step2.reweight.npy")],
      #   ["Herwig", os.path.join(theory_variation_dir, "39912440_1/model_weights_cc44b19d/Reweight_Step2.reweight.npy")],
      #   ["Sherpa", os.path.join(theory_variation_dir, "39912440_2/model_weights_afd3a072/Reweight_Step2.reweight.npy")]
      # ]

      # ensembled 15 trainings on nersc
      theory_variation_dir = "/home/badea/e+e-/aleph/UnfoldThrustResults/theory_reweighting/training-bf3b5fc3/"
      theory_variations = [
        ["Pythia8", os.path.join(theory_variation_dir, "Reweight_Step2_Ensemble_Pythia8.npy")],
        ["Herwig", os.path.join(theory_variation_dir, "Reweight_Step2_Ensemble_Herwig.npy")],
        ["Sherpa", os.path.join(theory_variation_dir, "Reweight_Step2_Ensemble_Sherpa.npy")],
      ]

      for i in range(n_systematics):
        for name, inFileName in theory_variations:
          temp = training_conf.copy()
          temp["job_type"] = f"TheoryUncertainty_{name}"
          temp["i_ensemble_per_omnifold"] = i
          temp["theory_variation_weights_path"] = inFileName
          confs.append(temp)

    # bootstrap mc
    # total_n_bootstraps_mc = 40
    # n_bootstraps_mc = math.ceil(total_n_bootstraps_mc / n_training_per_node)
    n_bootstraps_mc = 5*N_trainings_per_ensemble
    if args.run_bootstrap_mc:
      for i in range(n_bootstraps_mc):
        temp = training_conf.copy()
        temp["job_type"] = "BootstrapMC"
        temp["i_ensemble_per_omnifold"] = i
        confs.append(temp)

    # bootstrap data
    # total_n_bootstraps_data = 40
    # n_bootstraps_data = math.ceil(total_n_bootstraps_data / n_training_per_node)
    n_bootstraps_data = 5*N_trainings_per_ensemble
    if args.run_bootstrap_data:
      for i in range(n_bootstraps_data):
        temp = training_conf.copy()
        temp["job_type"] = "BootstrapData"
        temp["i_ensemble_per_omnifold"] = i
        confs.append(temp)
    
    # add configurations for ensembling
    # total_n_ensembles = 10 # 1 nominal + 10 ensembles = 11, N=10 -> 11*10 = 110 trainings 
    # n_ensembles = math.ceil(total_n_ensembles / n_training_per_node)
    n_ensembles = 10*N_trainings_per_ensemble
    if args.run_ensembling:
      for i in range(n_ensembles):
        temp = training_conf.copy()
        temp["job_type"] = "Ensembling"
        temp["i_ensemble_per_omnifold"] = i
        confs.append(temp)

    # add configuration for closure check with a single job
    if args.run_closure_test:
      temp = training_conf.copy()
      temp["run_closure_test"] = args.run_closure_test
      temp["job_type"] = "ClosureTest"
      confs.append(temp)

    # add configurations for hyperparameter scan
    if args.run_hyperparameter_scan:
      h_layer_sizes = [
         [50]*3, [100]*3, [200]*3, # scan the width
         [100]*2, [100]*4 # scan the depth
      ]
      h_batch_sizes = [256, 512, 1024, 2048]
      h_learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]
      hyperparameter_scan = list(itertools.product(h_layer_sizes, h_batch_sizes, h_learning_rates))
      for layer_sizes, batch_size, lr in hyperparameter_scan:
        temp = training_conf.copy()
        temp["job_type"] = "HyperparameterScan"
        temp["layer_sizes"] = layer_sizes
        temp["batch_size"] = batch_size
        temp["lr"] = lr
        temp["niter"] = 1
        confs.append(temp)

    # add configurations for niter scan
    if args.run_niter_scan:
      niter = list(range(1,7))
      for niter in niter:
        temp = training_conf.copy()
        temp["job_type"] = "NiterScan"
        temp["niter"] = niter
        confs.append(temp)

    # if no slurm config file provided then just launch job
    if args.slurm == None:
      
      print("No slurm config file provided. Running jobs locally.")
      for iC, conf in enumerate(confs):
          # only launch a single job
          if args.njobs != -1 and (iC+1) > args.njobs:
              continue
          train(conf)
    
    # if slurm config file provided then launch job on slurm
    else:
      
      # read in query
      query_path = os.path.abspath(args.slurm)
      if not os.path.exists(query_path):
        raise ValueError(f"Could not locate {args.slurm}")
      with open(query_path) as f:
        query = json.load(f)

      # submission
      executor = submitit.AutoExecutor(folder=top_dir)
      executor.update_parameters(**query.get("slurm", {}))
      # the following line tells the scheduler to only run at most 2 jobs at once. By default, this is several hundreds
      # executor.update_parameters(slurm_array_parallelism=2)
      
      # loop over configurations
      jobs = []
      with executor.batch():
          for iC, conf in enumerate(confs):
              
              # only launch a single job
              if args.njobs != -1 and (iC+1) > args.njobs:
                  continue
              
              # print(conf)

              job = executor.submit(train, conf) # **conf
              jobs.append(job)
