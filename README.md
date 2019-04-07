## POD+K for Real-Time Aeroelastic Pre-design Problem

Author: Oriol CHANDRE-VILA.

Advisors: Joseph MORLIER (ISAE-SUPAERO) and Sylvain DUBREUIL (ONERA). 

ISAE SUPAERO and ONERA, 2019.

### Presentation

This project aims to present the current work developed during my one-year research project within the umbrella of the Parcours Recherche proposed by ISAE-SUPAERO. This project has been monitored by Joseph MORLIER (ISAE) and Sylvain DUBREUIL (ONERA). Divided into three stages, first of all, some techniques have been tried in a 1-equation problem. Then the best methodology has been applied to a 2-equation problem and, finally, introducing the technique to more general problems has been approached.
The code is implemented with Python 2.7.


### Projects Overview and Objectives
This project has three main stages that have been performed during the last 12 months:

- A general approach to Reduced Order Models (ROM) [GitHub](https://github.com/mid2SUPAERO/PGD-ROM_CHANDRE).
- Application of the chosen methodology to an own code of fluid-structure interactions (internship in ONERA) [GitHub](https://github.com/mid2SUPAERO/POD-K_v1-CHANDRE).
- An updating of the previous code with a convergence of the reduced problem inside the Greedy ALgorithm is introduced. Besides, one could create more Kriging points thanks to the calculated Reduced Bases. This point is the one presented in this GitHub.

### Methodology

- OFFLINE PHASE: In this stage, a Greedy Algorithm has been used to create the Reduced Base POD for each parameter of interest; circulation in Aerodynamics and displacement in Structures. In each iteration (total number of iterations is predetermined as a variable), a finite number of points (second predetermined variable) is performed. The goal of the Greedy algorithm is to perform the reduced approach and compute an Error Indicator (EI). Then, there where this EI is the highest, run a complete analysis and add the results to the corresponding RB.
- ONLINE PHASE: Here, a Kriging interpolation (fed within the Offline Phase) is used in order to run only a specific case. With the Kriging, the values of displacement and circulation are calculated. What is really interesting (thanks to the Kriging) is the fact that the user will know the error of interpolation at the point. So, if one considers a point as interesting but the error is too big, the real computation could be run.

### In this GitHub...

1. Code: One can find all the codes needed to perform the study. Just several things to be commented:

    - The Python version used is the 2.7
    
    - Two packages need to be installed: [pyDOE](https://anaconda.org/conda-forge/pydoe) and [Scikit learn](https://anaconda.org/anaconda/scikit-learn)

    - A mesh creator (and the viewer) is used: [GMSH 3.0.6](https://gitlab.onelab.info/gmsh/gmsh/tags), please note that it is not the latest version available!
    
    - The two main scripts are the [Offline Phase Script](https://github.com/mid2SUPAERO/POD-K_v2-CHANDRE/blob/master/Codes/aero_struct/study/Offline_aerostruct.py) and the [Online Phase Script](https://github.com/mid2SUPAERO/POD-K_v2-CHANDRE/blob/master/Codes/aero_struct/study/Online_aerostruct.py)

2. Presentation: the support for the presentation can be found inside.

3. Report: The document where the project's results are explained is attached.

### Conclusions

(From the project report)

To conclude, we can disaggregate two error nature: one due to the action of Reduction the Problem (we are not including information that is part of the solution) and another one due to the interpolation method (which thanks to the Kriging Model can be estimated).

To improve the first source of errors, we identify the importance of the number of samples. For improving the results, we could augment the order of the Reduced Problem, we could reduce the problem (which is not really interesting in a Preliminary Stage) or substituting the fixed-point iteration method of the Greedy Algorithm.

To improve the Interpolation Error, we could augment the number of candidates. This added to a higher number of samples will results in more point to the Kriging and more points analysed while building the Reduced Base. Obviously, we could also enlarge the number of Kriging points.

Finally, the gain in time, as said, it is clear. And we could imagine a situation that this application could really be useful. If we imagine a room with Aerodynamic and Mechanical Engineers, the solutions and possible discussions could be evaluated in real time, allowing the Preliminary to progress faster. Thus, it will be interesting to be applied in a problem if only a lot of cases must be simulated. If not, it does not worth to perform such an expensive offline calculus.
