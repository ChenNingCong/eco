setopt rm_star_silent
rm runs/* -rf
rm model/* -rf
python eco_ppo.py --num_players 3 --track --wandb_project_name r-eco --cuda