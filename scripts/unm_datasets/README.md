These datasets were used in the 1998 paper called "Detecting Intrusions Using System Calls: Alternative Data Models".

https://www.cs.unm.edu/~immsec/systemcalls.htm

Files are gzip-ed (which drastically decreases size, e.g. 110MB was compressed to 330KB in one case). But ".gz" files can be read by pandas easily (without the need to convert) like:  
```python
df = pd.read_csv('file.gz', header=None, delimiter='\s+', engine='python') # python engine must be specified when delimiter is regex pattern
```

All files seem to have a common format. All have 2 columns (first = process/stream ID, second = type of system call).  
It looks like there's a little inconsistency in how the authors of the paper used files. In most cases the number of "intrusions" in the paper, was equal to number to process/stream IDs found in files (not equal to the number of files themselves). That was the case for `login`, `ps`, `inetd`, `stide` where files contained various PIDs (probably because a single program had multiple processes?), however in case of `named`, the number of "intrusions" was equal to 2 (number of files instead of PIDs).  

```
lpr_mit abnormal_files=1001(1001 pids), normal_files=2704(2703 pids)
lpr_unm abnormal_files=1001(1001 pids), normal_files=4298(4298 pids)
named abnormal_files=2(5 pids), normal_files=1(27 pids)
xlock abnormal_files=2(2 pids), normal_files=71(71 pids)
login abnormal_files=2(13 pids), normal_files=2(24 pids)
ps abnormal_files=2(26 pids), normal_files=2(24 pids)
inetd abnormal_files=1(31 pids), normal_files=1(3 pids)
stide abnormal_files=1(105 pids), normal_files=13726(13726 pids)
sendmail abnormal_files=0(0 pids), normal_files=71767(71760 pids)
```

1 file from lpr_mit normal dataset and 7 files from sendmail normal dataset were deleted (moved to "deleted" folder) because they were empty. Judging by the number of examples in the "Alternative Data Models" paper, the same was done there. Example empty file:  
```
lpr_mit/normal/857428790-sma-03Mar1997-171345-1212.log.int
```
