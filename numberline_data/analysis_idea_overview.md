# Network Analysis as in Wang 2022


https://github.com/aestrivex/bctpy/blob/master/bct/algorithms/centrality.py

Wang2020
Average `participation coefficient` PC for all nodes in each module was calculated to test which module drives the developmental-induced segregation effects. Further, `intra-connections` of each module and `inter-connections` between any two modules were calculated.


### Study details

- 54 kids, median-split by age; then 2 scans 2 years apart
    - younger: 10 - 12
    - older: 12 - 14
- multiplication task in scanner
    - relevant time points of the multiplication trials were extracted and concatenated over trials (control condition data not used?!)
- 142 ROIs  - 5 functional modules (Dosenbach, 2010)
    - 1 DMN; (2) FPN; (3) somato- motor network (SMN); (4) visual network (VN); (5) cingulo-opercular network (CON) 
    - ROIs (6-mm radius spheres) were generated using 3dcalc with AFNI (Cox, 1996) with the centre coordinates from the above functional template, and the average time series of each ROI were extracted
- scarcity threshold of 15% to remove weak brain connections and keep strong connections
- measures on connectivity matrix (Pearson correlation coefficients)
    - participation coefficient (PC) ( = 1 - proportion of intra-module connections | for each node)
    - intra-connections of each module 
    - inter-connections between all modules
    
#### Findings:
The results showed that the 
- default-mode (DMN) and frontal- parietal networks (FPN) became increasingly segregated over time. 
    - intra-connectivity within the DMN and FPN increased significantly with age,
    - inter-connectivity between the DMN and visual network decreased significantly with age. !!?!!
* (Such developmental changes were mainly observed in the younger children but not in the older children) 

- change in network segregation of the DMN was positively correlated with longitudinal gain in arithmetic performance in the younger children, 
- individual difference in network segregation of the FPN was positively correlated with arithmetic performance at Time 2 in the older children. 


--> decrease in PC = increase in intra-module connectivity ?!


### Karin's Hypothesis

1.  Differences in network segregation = participation coefficient (PC) (Wang et al., 2022).
    - Inter-module: Reduced inter-module segregation of task-based networks during numerical order fMRI-task in DD compared to TD children.
    - Intra-module: 
        - Increased intra-module connections in DD compared to TD children
        - Lower intra-SAL-module connections in DD leads to lower inter-module connections between DMN and FPN.
2. Correlations between graph measures and numerical performance.
3. Effects of training on functional segregation:

