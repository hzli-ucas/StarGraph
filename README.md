# StarGraph for [OGB WikiKG 2](https://ogb.stanford.edu/docs/linkprop/#ogbl-wikikg2)

Conventional representation learning algorithms for knowledge graphs (KG) map each entity to a unique embedding vector, ignoring the rich information contained in neighbor entities. We propose a method named StarGraph, which gives a novel way to utilize the neighborhood information for largescale knowledge graphs to get better entity representations. The core idea is to divide the neighborhood information into different levels for sampling and processing, where the generalized coarse-grained information and unique fine-grained information are combined to generate an efficient subgraph for each node. In addition, a self-attention network is proposed to process the subgraphs and get the entity representations, which are used to replace the entity embeddings in conventional methods. The proposed method achieves the best results on the ogbl-wikikg2 dataset, which validates the effectiveness of it.

|Method|Test MRR|Validation MRR|#Params|
|-|-|-|-|
|**StarGraph + TripleRE'**|**0.7201 ± 0.0011**|**0.7288 ± 0.0008**|86,762,146|
|TranS + NodePiece|0.6939 ± 0.0011|0.7058 ± 0.0018|38,430,804|
|TripleRE + NodePiece|0.6866 ± 0.0014|0.6955 ± 0.0008|36,421,802|
|InterHT + NodePiece|0.6779 ± 0.0018|0.6893 ± 0.0015|19,215,402|
|ComplEx-RP (50dim)|0.6392 ± 0.0045|0.6561 ± 0.0070|250,167,400|
|NodePiece + AutoSF|0.5703 ± 0.0035|0.5806 ± 0.0047|6,860,602|
|AutoSF|0.5458 ± 0.0052|0.5510 ± 0.0063|500,227,800|
|PairRE (200dim)|0.5208 ± 0.0027|0.5423 ± 0.0020|500,334,800|
|RotatE (250dim)|0.4332 ± 0.0025|0.4353 ± 0.0028|1,250,435,750|
|TransE (500dim)|0.4256 ± 0.0030|0.4272 ± 0.0030|1,250,569,500|
|ComplEx (250dim)|0.4027 ± 0.0027|0.3759 ± 0.0016|1,250,569,500|


+ This is the code to run StarGraph on the OGB WikiKG 2 dataset. 
Part of the code is based on [NodePiece repo](https://github.com/migalkin/NodePiece/tree/main/ogb).
+ A more comprehensive description of the method can be found at [StarGraph: A Coarse-to-Fine Representation Method for Large-Scale Knowledge Graph](https://arxiv.org/abs/2205.14209)

## Running
1. Install the requirements from the `requirements.txt`
2. Prepare the file storing the subgraphs as follows:  
&emsp;&emsp; a. Download the file of anchors using the `download.sh` script, provided by [NodePiece](https://github.com/migalkin/NodePiece/blob/main/ogb/download.sh)  
&emsp;&emsp; b. Generate the file of neighbors by running `python create_nborfile.py`
3. Run the `run_ogb.sh` script to reproduce the results of **StarGraph + TripleRE'** reported above

## Citation
If you find this work useful, please consider citing the paper:
```
@misc{li2022stargraph,
      title={StarGraph: A Coarse-to-Fine Representation Method for Large-Scale Knowledge Graph}, 
      author={Hongzhu Li and Xiangrui Gao and Yafeng Deng},
      year={2022},
      eprint={2205.14209},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
