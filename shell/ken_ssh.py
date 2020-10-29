#! /usr/bin/env python

import sys
import subprocess

server_list = {
    '439': {
        'user': 'web_server',
        'host': 'st-dz-rs439.yz',
        'path': '/home/web_server',
        'desc': 'KML Ceph'
    },
    '180': {
        'user': 'web_server',
        'host': 'bjpg-g180.yz02',
        'path': '/home/web_server',
        'desc': 'KCS Ceph MMU_GPU_MODEL'
    },
    '941': {
        'user': 'web_server',
        'host': 'st-dz-rs941.yz',
        'path': '/data/web_server/project/lijiahong',
        'desc': 'Data exporter tools, Model client, Ceph'
    },
    '948': {
        'user': 'web_server',
        'host': 'st-dz-rs948.yz',
        'path': '/data/web_server/project/lijiahong',
        'desc': 'Data exporter tools, Model client'
    },
    '66': {
        'user': 'mmu',
        'host': 'bjlt-rs66.sy',
        'path': '/home/mmu',
        'desc': 'Export Hive Data, Hadoop/Spark'
    },
    '110': {
        'user': 'web_server',
        'host': 'bjlt-hg110.sy',
        'path': '/home/web_server',
        'desc': 'GPU, Hadoop, Ceph'
    },
    '888': {
        'user': 'web_server',
        'host': 'sd-bjpg-rs888.yz02',
        'path': '/data/web_server',
        'desc': 'LabelDirRoot'
    },
    '9209': {
        'user': 'web_server',
        'host': 'bjfk-rs9209.yz02',
        'path': '/data/web_server/project/suiyao/combo_search',
        'desc': 'ComboSearch'
    },
    '1075': {
        'user': 'web_server',
        'host': 'bjpg-rs1075.yz02',
        'path': '/home/web_server/suiyao/topk',
        'desc': 'VideoSearch'
    },
    'zynn': {
        'user': 'web_server',
        'host': 'xlab-x6-mmu1.gcpeast1.useast.kwaidc.com',
        'path': '/home/web_server',
        'desc': 'ZYNN USEAST'
    },
    'zynn_gpu5': {
        'user': 'web_server',
        'host': 'xlab-x6-mmu5.gcpeast1.useast.kwaidc.com',
        'path': '/home/web_server',
        'desc': 'ZYNN GPU_5'
    },
    'zynn_gpu6': {
        'user': 'web_server',
        'host': 'xlab-x6-mmu6.gcpeast1.useast.kwaidc.com',
        'path': '/home/web_server',
        'desc': 'ZYNN GPU_6'
    },
}

def get_host(alias):
    server = server_list[alias]
    return server['user'] + '@' + server['host']

def get_path(alias, path):
    server = server_list[alias]
    return server['user'] + '@' + server['host'] + ':' + path

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_servers():
    columns = ['alias', 'user', 'host', 'path', 'desc']
    widths = [5, 11, 15, 36, 40]

    for column, width in zip(columns, widths):
        print bcolors.OKBLUE + ('{:' + str(width) + '}').format(column),
    print

    for alias in server_list:
        server = server_list[alias]
        print bcolors.WARNING + ('{:' + str(widths[0]) + '}').format(alias),
        for column, width in zip(columns[1:], widths[1:]):
            print bcolors.OKGREEN + ('{:' + str(width) + '}').format(server[column]),
        print 
    print bcolors.ENDC

def print_usage():
    print bcolors.WARNING + 'usage:'
    print bcolors.OKGREEN + '  ssh_tool.py ' + bcolors.OKBLUE + 'go alias'
    print bcolors.OKGREEN + '  ssh_tool.py ' + bcolors.OKBLUE + 'upload local_path dst_alias:dst_path'
    print bcolors.OKGREEN + '  ssh_tool.py ' + bcolors.OKBLUE + 'download src_alias:src_path dst_path'
    print bcolors.ENDC

try:
    task = sys.argv[1]
    if task == 'go':
        alias = sys.argv[2]
        args = [
            'ssh',
            get_host(alias),
            '-o', 'LocalCommand="cd {0}"'.format(server_list[alias]['path'])
        ]
    else:
        args = ['rsync', '-avC', '--progress']
        if task == 'upload':
            src_path = sys.argv[2]
            dst_alias, dst_path = sys.argv[3].split(':')
            args += [src_path, get_path(dst_alias, dst_path)]
        elif task == 'download':
            src_alias, src_path = sys.argv[2].split(':')
            dst_path = sys.argv[3] 
            args += [get_path(src_alias, src_path), dst_path]
    print args
    subprocess.call(args)
except:
    print_usage()
    print_servers()

