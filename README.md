**GOAL OF THE PROJECT**

Following the work of Yicheng Wu and Ashok Veeraraghavan presented in Nature LSA, this project is a continuation of the WISH algorithm for high resolution wavefront sensing. This would in term allow full field measurement in a hot atomic vapor experiment. We very sincerely thank Yicheng for sharing his Matlab code with us !

The approach presented in the 2 references is to use the SLM to modulate the face, and then use phase retrieval algos (Gerchberg-Saxton). The main advantage is that instead of doing one single calculation through a phase retrieval algo that is not perfect, one can average over the modulation thus greatly improving the precision of the reconstruction. 
![Principle of the algorithm](/images/wish_fig_2.png)
**COMPUTE CGH**

A small holographic pattern generator is available [here](ComputeCGH/compute_cgh.py). A detailed readme file explains the usage of the code.

**WISH**

In the [WISH](WISH/) folder you will find our python implementation of Yicheng Wu's Matlab code allowing replication of the results of their LSA paper. In the [dev](dev/) folder you will find the development version of our tailored version of this code. It includes features developed for the CGH generator such as automatic variable padding for the Gerchberg-Saxton loop.

**CONTRIBUTION**

If you want to contribute, you are more than welcome to do so. Simply **create a new branch with you name on it**. If you do not know how to do it, simply use the GitHub webpage and click on "branches" and type the name of the branch you want to create:
![How do I create a new branch ?](/images/create_branch.png)
**Then do not forget to checkout as this new branch !** :)
Also feel free to declare issues if you see some, the code is still far from ideal.

**PYTHON LIBS AND DEPENDENCIES** :  

In order to simulate light propagation, I use this library : 

https://github.com/opticspy/lightpipes 

It has already quite a lot of built in optical elements. 
Dependencies are [here](setup.py)

**REFERENCES** : 

https://hal.archives-ouvertes.fr/hal-00533180/document 

https://www.nature.com/articles/s41377-019-0154-x 

http://arxiv.org/abs/1711.01176 
