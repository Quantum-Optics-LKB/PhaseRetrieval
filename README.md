**GOAL OF THE PROJECT**

Following the 2 references below and taking advantage of my history with SLM's, the idea is to build a Michelson interferometer where one of the mirrors is a SLM. Then, we intend to find the optical contact not by tuning the lengths of the arms, but rather by tuning the phase of the SLM : when the exact "negative" phase of the input light is displayed, the intensity cancels completely and we are at the optical contact. Directly cancelling the phase is the brute force approach, and is not very straightforward a priori. The problem is that once the phase map is discretized by the SLM, the problem becomes infinitely degenerate because physically, there is no difference between a pixel at 1 rad, or 1 rad + k*2pi whatever the k (which is obviously not the case for a continuous phase map). 

The approach presented in the 2 references is to use the SLM to modulate the face, and then use phase retrieval algos (Gerchberg-Saxton). The main advantage is that instead of doing one single calculation through a phase retrieval algo that is not perfect, one can average over the modulation thus greatly improving the precision of the reconstruction. 
![Principle of the algorithm](/images/wish_fig_2.png)
**COMPUTE CGH**

For now, a holographic pattern generator is available [here](ComputeCGH/compute_cgh.py). The wavefront sensor is still in developpment.

**HOW DO I GET IT ?**

Start by pulling the repository. For this there are two options :
* Download the archive directly from GitHub.
![Which button to download the repository ?](/images/download_repo.png)
* Use the following commands :
Clone repository in your desired folder
```console
toto@pcdetoto:~/PathtotheplaceIwant/$ git clone https://github.com/quantumopticslkb/phase_retrieval.git
```
You will need to enter the login and the password of the GitHub account.

**USAGE**

The program runs in command line for now. Here is the syntax :
```console
toto@pcdetoto:~/PathtotheplaceIwant/$ python3 compute_cgh.py I IO cfg [-h][-phi0] [-output] [-mask_sr] [-s]
```
There are 3 positional arguments :
* `I` : The target intensity
* `I0` : The source intensity
* `cfg` : A config file containing the various physical parameters needed for the propagation. The template for such a file can be found [here](ComputeCGH/cgh_conf.conf)

The structure of the configuration file is the following :
```python
[params]
#size of the longest side of the source intensity in m
size_SLM = 3.2768e-3
#size of target intensity in m
size = 3.2768e-3
#wavelength of the field
wavelength = 532e-9
#propagation distance in m
z = 25e-3
#number of iterations for the GS loop
N_gs = 2
#modulation number (1 is disabled).
N_mod = 2
#modulation intensity (between 0 and 1) : how strong the initial phase is modulated
mod_intensity=0.1
#number of SLM levels (usually it's always 8 bits so 256 levels)
SLM_levels = 256
#Threshold above which the signal is considered
mask_threshold = 5e-2

[setup]
#Elements of the optical setup
#Attributes are the parameters from LightPipes, for instance f for a lens
#distance is the distance from the previous optical element
L1 = {'Name' : 'L1', 'Type' : 'Lens','Attributes' : 10e-2, 'distance' : 10e-2}
L2 = {'Name' : 'L2', 'Type' : 'Lens','Attributes' : 20e-2, 'distance' : 20e-2}

```

**For sampling reasons, the optimal propagation distance is between z and 4z where z = N dx^2 / lambda (where dx is the SLM pixel pitch**

For now the optical setup is not yet functionnal : the algo assumes a free space propagation from the SLM to the image plane.

There are 5 optional arguments:
* `-h` : Print the help string and exit
* `-phi0` : Path to initial phase of the source, for instance the calibration of the SLM that displays the hologram
* `-output` : Path to output folder in which the program will save the input images as well as the results
* `-mask_sr` : Path to a user defined mask
* `-s` : Silent option. If provided the program does not plot anything and just prompts progress

**The programm understands both absolute and relative paths but it is simplest to put the images you are going to use in the same folder as `compute_cgh.py`**

For Fourier computations reasons, the images must contain a "border" of zero to avoid meeting the reflective boundaries. If the provided images do not contain such a border, they will be padded up to twice their size.

The programm then outputs the results in a folder named "results_{time}" where time is the time at which the code ran. If an `-output` path was specified, the results will be written here. The code outputs `I`, `I0` (as PNG) and the calculated phase map / intensity map (as PNG). It also outputs a file named `rms_intensity.txt` which contains the RMS between the target image and propagated intensity map.

**Example :** The example can be downloaded [here](/examples/anti_ring)
The target intensity and source intensities are the following :
![anti_ring](/images/I_anti_ring_big.bmp)
![source_phase](/images/I0_512_big.bmp)

Now run the command :
```console
toto@pcdetoto:~/PathtotheplaceIwant/$ python3 compute_cgh.py I_anti_ring_big.bmp I0_512_big.bmp cgh_conf.conf -output results
```

The program should plot the following image of the auto defined mask for you to check :
![plot_sr](/images/plot_sr.png)

This mask will be the signal region where the intensity is imposed at each GS iteration.

After closing the plot, it will then run displaying a progress bar until it plots the final results like so :
![plot_result](/images/plot_result.png)

The recovered phase, target intensity, propagated intensity (recovered phase propagated to the image plane), the propagated phase as well as a cut of the propagated phase along the x axis. Note that the RMS between the reconstructed intensity and the target intensity and the conversion efficiency are displayed in the top left corner.
The `anti_ring` folder should now contain a subfolder named `results` with the images saved as PNG, a text file `metrics.txt` containing the metrics of the image : correlation coefficient between target intensity and final intensity, RMS and conversion efficiency. If you chose to run several modulations (i.e the GS loop starts with `N_mod` random phases and then averages the recovered phase), it will also contain the `N_mod` recovered phases saved as a numpy array `Phi.npy`.


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
