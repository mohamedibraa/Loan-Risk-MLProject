#!/bin/python3

import math
import os
import random
import re
import sys


if __name__ == '__main__':
    n = int(input().strip())
    st=''
    while n!=0:
        st+=str(n%2)
        n=int(n/2)
    for i in range(len(st),-1,-1):
        v=''
        for _ in range(i):
            v+='1'
        if v in st:
            print(i)
            break