# Senior Honours 2025
All the code I needed to carry out my senior honours project 'Finding the Missing Baryons with NASA's Next Generation Telescope' supervised by Prof Romeel Dave at the University of Edinburgh. 

Environment in which the code was tested out to run best: 

'sh_env'
python=3.9 yt=4.2.1 matplotlib=3.4.3 numpy=1.22 scipy=1.9

## Run instructions

For full data generation and analysis routine run:

bash full_analysis.sh MODEL WIND SNAP NLOS GAL_NO

Which just wraps up the get_results and get_data for all galaxies in the sample generated.

NLOS = currently modifies the range of angles at which the lines of sight are taken. 

## Acknowledgements

This repository is based on code originally developed by Sarah Appleby as part of her PhD project under the supervision of Romeel Dave at The University of Edinburgh.  

The original version can be found [here](https://github.com/sarahappleby/cgm/tree/master/absorption/ml_project).  

This fork has been adapted by Matylda Rejus i.e. myself for use in my Senior Honours project. I'm trying to keep track of changes I make as I go along via commit messages :)



