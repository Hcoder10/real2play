"""Wait for a bounded fresh-trace snapshot, then launch the Z1 trainer once."""
import _bootstrap
import argparse
from pathlib import Path
import sys
import time

def main():
    p=argparse.ArgumentParser();p.add_argument('--minimum',type=int,default=24);p.add_argument('--traces',required=True)
    p.add_argument('--done-report',help='Wait for the completed collection report before taking the snapshot')
    a,rest=p.parse_known_args()
    while (len(list(Path(a.traces).glob('*.jsonl')))<a.minimum
           or (a.done_report and not Path(a.done_report).exists())):time.sleep(2)
    import z1_train
    sys.argv=['z1_train','--traces',a.traces,*rest]
    z1_train.main()

if __name__=='__main__':main()
