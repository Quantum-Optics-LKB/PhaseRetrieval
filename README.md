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
toto@pcdetoto:~/PathtotheplaceIwant/$ git clone https://github.com/quantumopticslkb/phase_retrieval.git
```
You will need to enter the login and the password of the GitHub account.
The program runs in command line for now. Here is the syntax :
```console
toto@pcdetoto:~/PathtotheplaceIwant/$ python compute_cgh.py I IO cfg [-h][-phi0] [-output] [-mask_sr] [-s]
```
There are 3 positional arguments :
* `I` : The target intensity
* `I0` : The source intensity
* `cfg` : A config file containing the various physical parameters needed for the propagation. The template for such a file can be found [here](ComputeCGH/cgh_conf.conf)

There are 5 optional arguments:
* `-h` : Print the help string and exit
* `-phi0` : Path to initial phase of the source, for instance the calibration of the SLM that displays the hologram
* `-output` : Path to output folder in which the program will save the input images as well as the results
* `-mask_sr` : Path to a user defined mask
* `-s` : Silent option. If provided the program does not plot anything and just prompts progress

**The programm understands both absolute and relative paths but it is simplest to put the images you are going to use in the same folder as `compute_cgh.py`**

The programm then outputs the results in a folder named "results_{time}" where time is the time at which the code ran. If an `-output` path was specified, the results will be written here. The code outputs `I`, `I0` (as PNG) and the calculated phase map / intensity map (as PNG). It also outputs a file named `rms_intensity.txt` which contains the RMS between the target image and propagated intensity map.

**Example :** The example can be downloaded [here](/examples/harambe)
The target intensity and initial phase are the following :
![Petitangepartitroptôt](/images/harambe_512.bmp)
![calib_phase](/images/calib_512.bmp)

Now run the command :
```console
toto@pcdetoto:~/PathtotheplaceIwant/$ python compute_cgh.py harambe_512.bmp I0.bmp cgh_conf.conf -phi0 calib_512.bmp -output results 
```
The program should plot the following image of the auto defined mask for you to check :

The `harambe` folder should now contain a subfolder named `results` with the following images :


**CONTRIBUTION**

If you want to contribute, you are more than welcome to do so. Simply **create a new branch with you name on it**. If you do not know how to do it, simply use the GitHub webpage and click on "branches" and type the name of the branch you want to create:
![How do I create a new branch ?](/images/create_branch.png)
**Then do not forget to checkout as this new branch !** :)

**PYTHON LIBS AND DEPENDENCIES** :  

In order to simulate light propagation, I use this library : 

https://github.com/opticspy/lightpipes 

It has already quite a lot of built in optical elements. 
Dependencies are [here](setup.py)

**REFERENCES** : 

https://hal.archives-ouvertes.fr/hal-00533180/document 

https://www.nature.com/articles/s41377-019-0154-x 

http://arxiv.org/abs/1711.01176 
