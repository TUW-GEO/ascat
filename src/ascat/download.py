import argparse
import sys
from ftplib import FTP
import getpass

#TODO: create general FTP and GET objects to pass queries to

def parse_main_args_download(args):
    parser = argparse.ArgumentParser(
        description='ASCAT download command line interface.',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-s', '--source', help='Download source, EUMETSAT or HSAF')#FIXME: too specific
    parser.add_argument('-u', '--username', help='Username')
    parser.add_argument('-pw','--password', help='Password')
    parser.add_argument('-pr','--product', help='Name of product')
    
    #TODO: error handling when parameter is not passed

    return parser.parse_args(args), parser

def main_download(cli_args):
    args, parser = parse_main_args_download(cli_args)
    password = args.password#FIXME: getpass.getpass()
    if args.source.upper() == 'EUMETSAT':
        download_hsaf(username=args.username,
                     password=password,
                     product=args.product)
    elif args.source.upper() == 'HSAF':
        pass


def download_hsaf(username, password, product):
    #product options=('h10', 'h13', 'h34')
    with FTP('ftphsaf.meteoam.it') as ftp:
        try:
            import pdb;pdb.set_trace()
            ftp.login(username, password)
            print('Connected')
        except:
            print("Username or Password is incorrect")
        path = product + '/' + product + '_cur_mon_data'
        ftp.cwd(path)
        import pdb;pdb.set_trace()

def download_eumetsat(username, password):
    pass

def run_download():
    main_download(sys.argv[1:])
    
if __name__ == "__main__":
    run_download()
