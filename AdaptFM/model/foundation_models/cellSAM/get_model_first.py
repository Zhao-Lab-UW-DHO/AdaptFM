"""First time cellsam runs it needs the access token to get the model
this is run once at the time of install so users have the model saved locally
"""
import argparse
from cellSAM import  get_model

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--access_token')
  args = parser.parse_args()
  
  export DEEPCELL_ACCESS_TOKEN=args.access_token
  model = get_model(model='cellsam_extra')

if __name__ == "__main__":
    main()
