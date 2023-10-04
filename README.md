# Physics informed neural network for phase-field modeling of fracture

## Phase-field modeling of fracture
Modeling fracture using the phase-field method involves solving for the vector-valued elastic field, <b>u</b>$(u_{1},u_{2})$ and the scalar-valued phase-field, c.

The Allen-Cahn Equation is used to describe the evolution of crack, where the continuous scalar parameter c is used totrack the fracture pattern. The cracked region is represented by c = 1 while the undamaged portion is given by c = 0. 

The Allen-Cahn equation is defined as: 

$$η\dot c+\frac{g}{l}c-2(1-c)ψ_{ela}-gl\nabla^2c=0  (η=0)(1)$$    

$$ψ_{ela}=\frac{1}{2}σ_{ij}ε_{ij}$$

where 

$$ε_{11}=u_{1,1}$$

$$ε_{22}=u_{2,2}$$

$$ε_{12}=\frac{1}{2}(u_{1,2}+u_{2,1})$$

$$σ_{11}=(\frac{2μG}{1-μ}(u_{1,1}+u_{2,2})+2Gu_{1,1})(1-c)^2$$

$$σ_{22}=(\frac{2μG}{1-μ}(u_{1,1}+u_{2,2})+2Gu_{2,2})(1-c)^2$$

$$σ_{12}=G(u_{1,2}+u_{2,1})(1-c)^2$$

$$G=\frac{E}{2(1+μ)}$$

The plane stress equations is defined as:

$$\begin{cases}
σ_{11,1}+σ_{12,2}=0\\
σ_{21,1}+σ_{22,2}=0
\end{cases}$$

Thus, the above equations can be expressed as:

$$\begin{cases}
((\frac{2}{1-μ})u_{1,11} + u_{1,22} + (\frac{1+μ}{1-μ})u_{2,21})(1-c)^2-4(1-c)[\frac{μ}{1-μ}(u_{1,1}+u_{2,2})+u_{1,1}]c_{,1}-2(1-c)(u_{1,2}+u_{2,1})c_{,2}= 0\\
((\frac{2}{1-μ})u_{2,22} + u_{2,11} + (\frac{1+μ}{1-μ})u_{1,12})(1-c)^2-4(1-c)[\frac{μ}{1-μ}(u_{1,1}+u_{2,2})+u_{2,2}]c_{,2}-2(1-c)(u_{1,2}+u_{2,1})c_{,1}= 0
\end{cases}$$

$t\in[0,1]$; $x_{1}\in[0,1]$; $x_{2}\in[0,1]$ 

To prevent cracks from healing:

$$\dot c ≥ 0$$

## Single-edge notched tension example
In this example, we consider a unit square plate with a horizontal crack from the midpoint of the left outer edge to the center of the plate.
The geometric setup and boundary conditions of the problem are shown in Fig. 1.

![image](https://github.com/QinghuiXiao/PF_PINN/assets/138593048/4f94e13a-3d36-449b-bbfc-3475cd2873c4)

$$Fig. 1. Single-edge~notch~tension~example.$$

The plate bottom is fixed, the plate top is fixed in the horizontal direction and prescribed an upward displacement in the vertical direction. There does not exist body force or traction force. The Young’s modulus is $E=210GPa$, the Poisson’s ratio is $μ=0.3$, the critical energy release rate is $g=0.0027 KN/mm$, and the length parameter is $l=0.01mm$, which are all taken from [1]. 

Additionally, some initial conditions and boundary conditions are known and can be used to help train the model.  

The boundary conditions are:

$u_{1}(t,x_{1},0)=0$  

$u_{2}(t,x_{1},0)=0$   

$u_{2}(t,x_{1},1)=at$  (a=0.001)

And this problem is subject to homogeneous Neumann boundary condition:

$u_{1,1}(t,0,x_{2})= u_{1,1}(t,1,x_{2})=0$ 

The initial conditions are:

The point at the crack: $c(0,x_{1}\in[0,0.5],x_{1}=0.5)=1$,Remove the points at the crack:$c(0,x_{1}\in[0,1],x_{2}(x_{2}\neq0.5))=0$.

  
[1]Li Z, Shen Y, Han F, et al. A phase field method for plane-stress fracture problems with tension-compression asymmetry[J]. Engineering Fracture Mechanics, 2021, 257: 107995.
