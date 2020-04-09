**GOAL OF THE PROJECT**

Following the 2 references below and taking advantage of my history with SLM's, the idea is to build a Michelson interferometer where one of the mirrors is a SLM. Then, we intend to find the optical contact not by tuning the lengths of the arms, but rather by tuning the phase of the SLM : when the exact "negative" phase of the input light is displayed, the intensity cancels completely and we are at the optical contact. Directly cancelling the phase is the brute force approach, and is not very straightforward a priori. The problem is that once the phase map is discretized by the SLM, the problem becomes infinitely degenerate because physically, there is no difference between a pixel at 1 rad, or 1 rad + k*2pi whatever the k (which is obviously not the case for a continuous phase map). 

The approach presented in the 2 references is to use the SLM to modulate the face, and then use phase retrieval algos (Gerchberg-Saxton). The main advantage is that instead of doing one single calculation through a phase retrieval algo that is not perfect, one can average over the modulation thus greatly improving the precision of the reconstruction. 
![Principle of the algorithm](/images/wish_fig_2.png)

**USAGE**

Start by pulling the repository. For this there are two options :
* Download the archive directly from GitHub.
![Which button to download the repository ?](/images/download_repo.png)
* Use the following commands :
Clone repository in your desired folder
```console
toto@pcdetoto:~/Documents/quantum_optics_repo/$ git clone https://github.com/quantumopticslkb/phase_retrieval.git
```
**CONTRIBUTION**

If you want to contribute, you are more than welcome to do so. Simply **create a new branch with you name on it**. If you do not know how to do it, simply use the GitHub webpage and click on "branches":

**PYTHON LIBS** :  

In order to simulate light propagation, I use this library : 

https://github.com/opticspy/lightpipes 

It has already quite a lot of built in optical elements. 

**REFERENCES** : 

https://hal.archives-ouvertes.fr/hal-00533180/document 

https://www.nature.com/articles/s41377-019-0154-x 

http://arxiv.org/abs/1711.01176 
