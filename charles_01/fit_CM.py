import argparse

print("hello")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('--bids_folder', default='/mnt_01/ds-ASD')
    parser.add_argument('--confspec', default='36P')
    parser.add_argument('--task', default='magjudge') # adjust
    parser.add_argument('--ses', default=1, type=int) # adjust
    cmd_args = parser.parse_args()
    
    main(cmd_args.subject, cmd_args.bids_folder,
        confspec=cmd_args.confspec,
        task=cmd_args.task,
        ses=cmd_args.ses)