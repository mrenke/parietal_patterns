

# Precision functional mapping

"Precision Functional Mapping of Individual Human Brains" - Gordon 2017
- Infomap algorithm for community detection


### “consensus” assignment procedure [Gordon 2014 - Cortical Area Parcellation]
with the goal of incorporating information both from more sparse thresholds, in which smaller networks were likely to emerge, and more dense thresholds, in which more parcels were likely to be successfully assigned. In this procedure, each node was given the community assignment it had at the sparsest possible threshold at which it was successfully assigned. The node assignments were “cleaned up” by removing small communities that were only present at one threshold. This procedure is nearly identical to the method used to collapse previously published voxel-wise community assignments (Power et al. 2011) across thresholds to create a single network map (the “Power communities” map described above). We note that this procedure does not attempt to comprehensively describe all features of the network, and may be especially poor at capturing nonhierarchical network features (which do occur infrequently). Rather, it provides a single, summary view of the brain’s networks.

- sparse thresholds: smaller networks were likely to emerge
- dense thresholds, in which more parcels were likely to be successfully assigned

### matching approach [Gordon 2017]
Putative network identities were then assigned to each subject’s communities (and to the communities from the average of the individual subjects) by matching them at each threshold to the above independent group networks. This matching approach proceeded as follows. At each density threshold, all identified communities within an individual were compared (using spatial overlap, quantified with the `Jaccard index`) to each one of the independent group networks in turn. The best-matching (highest-overlap) community was assigned that network identity; that community was not considered for comparison with other networks within that threshold. Matches lower than Jaccard = 0.1 were not considered (to avoid matching based on only a few vertices). Matches were first made with the large, well-known networks (Default, Lateral Visual, Motor, Cingulo-Opercular, Fronto-Parietal, Dorsal Atten- tion), and then to the smaller, less well-known networks (Ventral Attention, Salience, Parietal Memory, Contextual Association, Medial Visual, Motor Foot).
‘consensus’’ network assignment was derived by collapsing assignments across thresholds, giving each node the assignment it had at the sparsest possible threshold at which it was successfully assigned to one of the known group networks
contiguous network pieces that were smaller than 30 mm2 were removed, following (Gordon et al., 2017a),

- Laumann 2015: System assignments were computed at a range of edge densities (0.05% to 5%)
A ‘‘consensus’’ assignment was derived by collapsing across thresholds as described in Gordon et al. (2014b), giving each node the assignment it has at the sparsest possible threshold at which it was suc- cessfully assigned

# OLD & Additional

"Individual Variation in Functional Topography of Association Networks in Youth" - Cui 2020 {Satterwhaite}
non-negative matrix factorization (NMF) (Lee and Seung, 1999) to derive individualized functional networks

"A precision functional atlas of personalized network topography and probabilities"
Custom code to generate individual-specific networks and probabilistic maps can be found at https://github.com/DCAN-Labs/compare_matrices_to_assign_networks (v1.0) and is publicly available. IM community detection code is available at www.mapequation.org (v1.4 was used in this manuscript)
MATLAB based though :( 


"Probabilistic mapping of human functional brain networks identifies regions of high group consensus"
Brain networks were identified in individual subjects by a winner-take-all procedure (similar to that employed in Gordon et al. (2017b)) which assigned each cortical gray matter voxel in a particular subject to one of 14 network templates.
https://www.sciencedirect.com/science/article/pii/S1053811921004419#sec0002


* https://github.com/edickie/ciftify - put preprocessed T1 and fMRI data into an HCP like folder structure |  making working with cifty format a little easier