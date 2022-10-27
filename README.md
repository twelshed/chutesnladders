# chutesnladders
Supradegnerate? No I'm normal degenerate.

setup:

 >git clone https://github.com/twelshed/chutesnladders.git
 
 >cd chutesnladders
 
 >git checkout -b awsready
 
 >git pull origin awsready

 >conda env create -f env.yml
 
 >conda activate snl


To run a single experiment:

#create a paramset

>python generate_paramset.py

#find a param json you want to run ex: paramset/iWab/cYFPnG.json

>python run.py --paramset paramset/iWab/cYFPnG.json

#results are in experiments/iWAb/cyFPnG

>python reconstruct_results.py --save_figs True --exp_id iWAb

To run many simulations (A whole batch):

#make a paramset as before then run:

# your batch_id may differ for this example it's 'iWAb'

>sh run.sh paramset/iWAb

# to reconstruct all sims in a batch run 

>python reconstruct_results.py --save_figs True --exp_id iWAb

Additional caveats:

General simulation parameters are in Config.py
    -n_step is the length of the simulation
    -N is the number of walks per step of the simulation per particle
    -n_parts is the number of particles
    -sigma is a scaling parameter on the length of any particular walk larger = long walk steps
