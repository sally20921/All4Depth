# Computing the Pixel Coordinates of a 3D Point
## Perspective Projection
### One of the Most Common Questions about 3D Rendering on the Web
* How do I find the 2D pixel coordinates of a 3D point? is one of the most common questions related to 3D rendering on the web.
* It is an important question indeed because it is the fundamental method by which an image of a 3D scene is formed.
* We will use the term rasterization to describe the process of finding 2D pixel coordinates of 3D points. 
* Rasterization in its broader sense, refers to the process of converting 3D shapes into a raster image. A raster image, as explained in the previous lesson, is the technical term given to a digital image; it designates a two dimensional array (or retangular grid if you prefer) of pixels.

* Don't be mistaken; different rendering techniques exist for producing images of 3D senes. Rasterization is only one of them. Ray-tracing is another. Note though that all these techniques rely on the same concept to produce that image: the concept of perspective projection. Therefore, for a given camera and a given 3D scene, all rendering techniques produce the same visual result; they just use a different approach to produce that result.

* Note also that computing the 2D pixel coordinates of 3D points, is only one of the two steps in the process of creating a photo-realistic image. The other step is the process of shading, in which the color of these points will be computed to simulate the appearance of objects. you need more than just converting 3D points to pixel coordinates to produce a complete image.

* To understand rasterization, you first need to be familiar with a series of important techniques which we will also introduce in this chapter. Read this lesson carefully, as it will provide you with the very basic tools almost all rendering techniques are built upon.

* We will use matrices in this lesson a lot so read the Geometry lesson first if you are not comfortable with coordinate systems and matrices yet.

* We will apply the techniques studied in this lesson to render a wireframe image of a 3D object (adjacent image). 

### A Quick Refersher on the Perspective Projection Process
* We talked about the perspective projection process in quite a few lessons already. In short, this technique can be used to create a 2D image of a 3D scene, by projecting points or vertices making up the objects of that scene, onto the surface of a canvas. Why are we doing that? Because this is more or less the way the human eye works, and since we are used to see the world through our eyes, it's quite natural to think that images created that way, will also look natural to us. You can see the human eye as just a point in space. What we see of the world is the result of light rays (reflected by objects), travelling to this point and entering the eye (the eye is obviously not exactly a point; it is an optical system converging rays onto a small surface-the retina). So again, one way of making an image of a 3D scene in CG is to do the same thing, which you can get by projecting vertices onto the surface of the canvas (or the surface of the screen) as if they were sliding along straight lines connecting the vertices themselves to the eye. 

- Figure 1. to create an image of a cube, we just need to extend lines from the objects corners towards the eye and find the intersection of these lines with a flat surface perpendicular to the line of sight.

It is important to understand that the perspective projection is just an arbitrary way of representing 3D geometry onto a two-dimensional surface. It is the most commonly used way because it simulates forshortening which is one of the most important properties of human vision: objects in the distance appear smallter than objects close by. Nonetheless, as mentioned in the Wikipedia article on perspective, it is important to understand that while creating realistic images, perspective stays an approximate representation, on a flat surface of an image as it is seen by the eye. THe important word here is approximate.

In the aforementioned lesson, we also explained how the world coordinates of a point located in front of the camera (and enclosed within the viewing frustum of the camera, thus visible to the camera), can be computed using a simple geometric construction based on one of the properties of similar triangles. We will review this technique one more time in this lesson. It turns out that the equations to compute the coordinates of a projected points can actually somehow be expressed in the form of a 4x4 matrix. If you don't use the matrix form, computing the projected point's coordinates is of course possible. It is in itself not very complex but requires nonetheless a series of operations on the original point's coordinates: this is what you will learn in this lesson. However, expressed in the form of a matrix, you can reduce this series of operations to a single point-matrix multiplication. Being able to represent this critical operation in such a compact and easy to use from is the main advantage of this approach. 

* It turns out, that the perspective projection process, and its associated equations, can be expressed in the form of a 4x4 matrix indeed, as we will demonstrate in lesson 5. This is what we call the perspective projection matrix. Multiplying any point whose coordinates are expressed with respect to the camera coordinate system by this perspective projection matrix will give you the position (or coordinate) of that point onto the canvas. 

- Figure 2. among all light rays reflected by an object, some of these rays enter the eye, and the images we have of this object, is the result of these rays.
- Figure 3: the projection process can be seen as if the point we want to project was moved down along a line connecting the point or the vertex itself to the eye. We can stop moving the point along that line when it lies on the plane of the canvas. Obviously we don't slide the point along this line explicitly, but this is how the projection process can be interpreted.

### Finding the 2D pixel coordinates of a 3D Point Explained from Beginning to End
* When a point or vertex is defined in the scene and is visible to the eye or to the camera, it appears in the image as a dot. We already talked about the perspective projection process which is used to convert the position of that point in 3D space to a position on the surface of the image. But this position is not expressed in terms of pixel coordinates. So how do we actually find the final 2D pixel coordinates of the projected poiotn in the image? 

### 4x4 Matrix Visualized as a Cartesian Coordinate System
* As you know, objects in 3D can be transformed using any of the three following operators: translation, rotation and scale. If you remember what we said in the lesson dedicated to Geometry, linear transformations (in other words any combination of any of these three operators) can be represented by a 4x4 matrix. Remember that the first three coefficients along the diagonal encode the scale, the first three values of the last row encode the translation and the 3x3 upper-left inner matrix encodes the rotation (the red, green and blue coefficients). 

* It might be difficult when you look at the coefficients of a matrix to know exactly what the scaling or rotation values are because rotation and scale are sort of combined within the first three coefficients along the diagonal of the matrix. So let's ignore the scale for now, and only focus on rotation and translation. As you can see we have nine coefficients that represent a rotation. But how can we interpret what these nine coefficients are? So far we looked at matrices. But let's consider what coordinate systems are and by connecting the two together- matrices and coordinate systems-we will answer this question.

* The only Cartesian coordinate system we talked about so far, is the world coordinate system. This coordinate system is a convention used to define the coordinates [0,0,0] in our 3D virtual space and three unit axes orthogonal to each other. It's the prime meridian of a 3D scene, a referenec to which any other point or any other arbitrary coordinate system is measured to. Once this coordinate system is defined, we can create other Cartesian coordinate systems and as with points, these coordinate systems are defined by a position in space (a translation value) but also by three unit axes or vectors orthogonal to each other (which by definition what Cartesian coordinate systems are). Both the position and the values of these three unit vectors are defined with respect to the world coordinate system as depicted in figure 4.

-Figure 4. coordinate systems translation and axes coordinates are defined with respect to the world coordinate system.

* In conclusion, we can say that a 4x4 matrix actually represents a coordinate system (or reciprocally that any Cartesian coordinate system can be represented by a 4x4 matrix). It's really important that you always see a 4x4 matrix as nothing else than a coordinate system and vice versa.


