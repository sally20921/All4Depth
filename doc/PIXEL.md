# Perspective Projection
* Let's quickly recall here what the perspective projection is. 
* In short, this technique can be used to create a 2D image of a 3D scene, by projecting points or vertices making up the objects of that scene, onto the surface of a canvas. 
* Why are we doing that? Because this is more or less the way the human eye works, and since we are used to see the world through our eye, it's quite natural to thinkg that images created that way, will also look natural to us.

* It is important to understand that the perspective projection is just an arbitrary way of representing 3D geometry on to a two-dimensional surface. 
* It is important to understand that while creating realistic images, perspective stays an approximate representation, on a flat surface, of an image as it is seen by the eye. 

* In the aforementioned lesson, we also explained how the world coordinates of a point located in front of the camera can be computed using a simple geometric construction base on one of the properties of similar triangles.

-Figure 2. among all light rays reflected by an object, some of these rays enter the eye, and the image we have of this object, is the result of these rays.

- Figure 3. The projection process can be seen as if the point we want to project was moved down along a line connecting the point or the vertex itself to the eye. We can stop moving the point along that line when it lies on the plane of the canvas. 


* It turns out that the equations to compute the coordinates of a projected points can actually somehow be expressed in the form of 4x4 matrix.
* It turns out, that the perspective projection process, and its associated equations, can be expressed in the form of a 4x4 matrix.
* Multiplying any point whose coordinates are expressed with respect to the camera coordinate system, by this perspective projection matrix, will give you the position (or coordinates) of that point onto the canvas.

* In this lesson, we will learn about computing the 2D pixel coordinates of a 3D point without using the perspective projection matrix. To do so, we will need to learn how we can project a 3D point onto the surface of a 2D drawable surface using some simple geometry rules. Once we understand the mathematics of this process, we will then be ready to study the construction and use of the perspective projection matrix, a matrix used to simplify the projection step. 

## Finding the 2D pixel coordinates of a 3D point Explained from Beginning to End
* We already talked about the perspective projection process which is used to convert the position of that point in 3D space to a position on the surface of the image. 
* But this position is not expressed in terms of pixel coordinates. 
* How do we actually find the final 2D pixel coordinates of the projected point in the image? 
 
### World Coordinate System and World Space
* When a point is first defined in the scene, we say its coordinates are defined in world space: the coordinates of this point are defined with respect to a global or world Cartesian coordinate system. The coordinate system has an origin, which is called the world origin and the coordinates of any point defined in that space, are defined with respect to that origin. Points are defined in world space. 

### 4x4 Matrix Visualized as a Cartesian Coordinate System
* Objects in 3D can be transformed using any of the three following operators: translation, rotation and scale. 
* Linear transformation (in other words any combination of any of these operators) can be represented by a 4x4 matrix.
* Remember that the first three coefficients along the diagonal encode the scale, the first three values of the last row encode the translation and the 3x3 upper-left inner matrix encodes the rotation.


* It might be difficult when you look at the coefficients of a matrix to know exactly what the scaling and rotation values are because rotation and scale are sort of combined within the first three coefficients along the diagonal of the matrix.
* So let's ignore scale for now, and only focus on rotation and translation. 
* As you can see we have nine coefficients that represent a rotation. But how can we interpret what these nine coefficients are? 

* Once this coordinate system is defined, we can create other Cartesian coordinate systems and as with points, these coordinate systems are defined by a position in space (a translation value) but alos by three unit axes or vectors orthogonal to each other. Both the position nad the values of these three unit vectors are defined with respect to the world coordinate system. 

* The upper-left 3x3 matrix of our 4x4 matrix is actually nothing else than the coordiantes of our arbitrary coordinate system's axes. We have three axes, each with three coordinates which makes nine coefficients. 

* In conclusion, we can say that a 4x4 matrix actually represents a coordinate system (or reciprocally that any Cartesian coordinate system can be represented by a 4x4 matrix). 
* It's really important that you always see a 4x4 matrix as nothing else than a coordinate system and vice versa (we also sometimes speak of a local coordinate system in reference to the global coordinate system which in our case, is the world coordinate system). 

### Local vs Global Coordinate System 
* Now that we have established how a 4x4 matrix can be interpreted (and introduced the concept of local coordinate system), let's recall what local coordinate systems are used for. 
* By default, the coordinates of a 3D point are defined with respect to the world coordinate system. The world coordinate system though, is just one among an infinity of possible coordinate systems. 

* When you move a 3D object in a scene such as a 3D cube, transformation applied to that object (translation, scale, rotation) can be represented by what we call a 4x4 transformation matrix. 
* This 4x4 transformation matrix can be seen as the object local frame of reference or local coordinate system. 
* In a way, you don't really transform the object, but transform the local coordinate system of that object, but since the vertices making up the objects are defined with respect to that local coordinate system, moving the coordinate system moves the object's vertices with it.

* It's important to understand that we don't explicitly transform the coordinate system. We translate, scale and rotate the object, these transformations are represented by a 4x4 matrix, and this matrix can be visualized as a coordinate system. 

## Transforming Points from One Coordinate System to Another
* As suggested before, it is sometimes more convenient to operate on points when they are defined with respect to a local coordinate system rather than 
