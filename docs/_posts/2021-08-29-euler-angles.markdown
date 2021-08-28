---
layout: post
title:  "3D Rotations and Euler Angles"
date:   2021-08-26 06:24:18 +0900
categories: vision
---

# 3D Rotations and Euler Angles

* In this article, we will see what Euler Angles are, how they are calculated, and how the rotational motion of a rigid body in three-dimensional Euclidean space can be calculated. 

## Euler's Theorem

> Each movement of a rigid body in three-dimensional space, with a point that remains fixed, is equivalent to a single rotation of the body around an axis around an axis passing through the fixed point

This theorem was formulated by Euler in 1775. In other words, if we consider two Cartesian reference systems, one $(X_0, Y_0, Z_0)$ and the other $(X_1, Y_1, Z_1)$ which have the same origin point $O$, but different orientation, there will always be a single axis of rotation which the first system will assume the same configuration as the second system. 

In an even simpler way, any rotation can be described by a sequence of three successive rotations, also called elementary rotations, which occur around one of the three coordinate axes $X, Y$ and $Z$. 

## The Elementary Rotations
There are therefore three elementary rotations, each around its Cartesian reference axis $X$, $Y$ and $Z$. But the rotation around an axis can occur in two opposite directions. But which of the two is the positive one?

Rotation around an axis is positive if it meets the right hand rule. For example, in the case of rotation around the $z$ axis, the rotation will be positive depending on the arrangement of $X$ and $Y$ axes in the representation. 

Each elementary rotation can be transcribed as a 3x3 matrix (homogeneous transformation).

* Rotation on the $X$ axis: $$\begin{pmatrix} 1&0&0\\0&\cos{\theta}&-\sin{\theta}\\0&\sin{\theta}&\cos{\theta}\end{pmatrix}$$

* Rotation on the $Y$ axis: $$\begin{pmatrix}\cos{\theta}&0&\sin{\theta}\\0&1&0\\-\sin{\theta}&0&\cos{\theta}\end{pmatrix}$$

* Rotation on the $Z$ axis: $$\begin{pmatrix} \cos{\theta}&-\sin{\theta}&0\\\sin{\theta}&\cos{\theta}&0\\0&0&1\end{pmatrix}$$

## Euler's Angles
Therefore a generic rotation is described in turn by a rotation matrix $R$. Any matrix of this type can be described as the product of successive rotations around the principal axes of $XYZ$ coordinates, taken in a precise order.

So any rotation could be decomposed into the sequence of three elementary matrices. For example, the most intuitive is that of which is obtained first by performing a rotation on the $X$ axis by an angle $\phi$, then on the $Y$ axis by an angle $\theta$ and finally on the $Z$ axis by an angle $\psi$. 

$$ R_x(\phi) \rightarrow R_y(\theta) \rightarrow R_z(\psi) $$

The triplet of the angles used in these three elementary rotations are the Euler Angles and are normally indicated by $(\phi, \theta, \psi)$. 

But the $XYZ$ rotation sequence is only one of 12 possible combinations. There are different possible combinations of three elementary rotations, such as $ZYX, ZYZ, XYX$, etc. Each of these will have a different convention for expressing Euler Angles. 

## The Rotation of a Point in Space
Euler transformations with their relative angles are a wonderful tool for applying rotations of points in space. The simplest example of application of what we have already seen in the article is the rotation of a point located in a coordinate space $(X,Y,Z)$. 

A point in space can be represented by a 3-element vector that characterizes its values on the three coordinate axes. If we want to apply a rotation at this point it will be sufficient to multiply this vector precisely with the rotation matrix and thus obtain another vector. 

## Limits of Euler's Representation 
The technique we have seen is baed on the use of a sequence of elementary rotations referring to one of the Cartesian axes at a time. By applying these rotations in sequence it can happen that one of the reference axes can collapse into another. For example with rotations of 90 degrees or 180 degrees.


Another drawback is that the angles depend on the sequence of the three rotations around the Cartesian axes and reported as the name of the convention: $ZYZ, XZX, YXZ$, etc. Each of these sequences gives a triplet of Euler angles with different values, as we have verified above. Then the rotation matrix and the inverse formula will change accordingly. 

## Conclusion 
Despite all these drawbacks, Euler Angles are widely used today and are a very important reference point for those who work in the field of CAD modeling, 3D video game engines, and robotics and automation in general. However, in practice, a more complex but more effective mathematical model is often used, the Hamilton quaternions. 
