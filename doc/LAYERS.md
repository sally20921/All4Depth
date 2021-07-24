# Geometry 
## Introduction to Geometry
* Points, vectors, and matrices are instrumental in the process of making CG images.
* A point is a position in a three-dimensional space.
* A vector, on the other hand, usually means direction (and some corresponding magnitude, or size) in three-dimensional space. 
* Vectors can be thought of arrows pointing various directions. 
* Sometimes it is necessary to add a fourth element for mathematical convenience. An example of a point with homogeneous coordinates is *P_H = (x,y,z,w)*. 
* Homogeneous points are used when it comes to multiplying points with matrices.

## A Quick Introduction to Transformations
* One of the most common operations we perform on points in CG consists of simply moving them around in space.
* This transformation is more specifically called translation and it plays a vital role in the rendering process.

* The translation operator is nothing more than a linear transformation of the original point (which can be viewed as an input position point).
* Applied to a vector (which, remember is a direction), translation has not meaning. 
* This is because where the vector begins (that is, where it is centered) is not important; regardless of the position, all arrows of the same length, pointing in the smae direction are equivalent.
* Instead, we very commonly use another linear transformation on vectors: rotation. 
* Let's just consider translation for points and rotations for vectors.
* *P -> Translate -> P_T*, *V -> Rotate -> V_T*.

* When the length of a vector is exactly 1, we say that the vector is normalized. 
* The act of normalizing a vector involves altering the vector such that its length becomes 1, but its direction remains unchanged.
* Most of the time, we will want our vectors to be normalized. 

* Imagine that you trace a line from point *A* to point *B*. The line created is a vector in the sense that it indicates where point *B* is located relative to point *A*. 
* That is, it gives the direction of *B* as if you were standing on *A*. The length of the vector in this case indicates the distance from *A* to *B*. 

* Normalization of vectors is often a source of bugs in applications everytime you declare a vector (or even use one), we recommend that you always consciously ask yourself if this vector is/isn't or should/shouldn't be normalized.

## Normals
* A normal is the technical term used in Computer Graphics (and Geometry) to describe the orientation of a surface of a geometric object at a point on that surface. 
* Technically, the surface normal to a surface at point *P*, can be seen as the vector perpendicular to a plane tangent to the surface at *P*. 
* Normals play an important role in shading where they are used to compute the brightness of objects.

* Normals can be thought of as vectors with one caveat: they do not transform the same way that vectors do. 

# Coordinate Systems
## Introducing Coordinate Systems
* In the previous chapter we mentioned that points and vectors are represented with three real numbers.
* But what do these numbers mean?
* Each number represents a singed distance from the origin of a line to the position of the point on that line. 
* For example, consider drawing a line and putting a mark in the middle. We will call this mark the origin. 
* This mark becomes our point of reference: the position from which we will measure the distance to any other point.

* In theory, the distance between two points on that line could be infinitely large. 
* In the world of computers, there is a practical limit to the value you can represent for a number.
* Thankfully this maximum value is usually big enough to build most of the 3D scenes we will want to render; all the values we deal with in the world of CG are bounded anyway.

* Now that we have a line and an origin we add some additional marks at a regular interval (unit length) on each side of the origin, effectively turning our line into a ruler.
* With the ruler established, we can simply use it to measure the coordinate of a point from the origin (coordinate being another way of saying the signed distance from the origin to the point).
* In computer graphics and mathmatics, the ruler defines what we call an axis.

* If the point we are interested is not on the axis, we can still find the point's coordinates by projecting it onto the ruler using a vertical line.
* The distance from the origin to the intersection of this vertical line with the ruler is the coordinate of that point with respect to that axis. 
* We have just learned to define the coordinate of a point along an axis.

## Dimensions and Cartesian Coordinate Systems
* Let's call the horizontal rule from before the x-axis. 
* We can draw another rule perpendicular to the x-axis as its origin.
* We will call this the y-axis.
* For any point, we can determine both the x- and y-coordinate by drawing perpendicular lines to each axis and measuring the distance from those intersections to the origin.

* We can now find two numbers, or two coordinates, for an arbitrary point: one for the x-axis, and one for the y-axis. 
* Thus, by placing two axes, we have defined a two dimensional space called a plane.

* For example, consider drawing a number of points on a piece of paper. This piece of paper occupies a two dimensional space, i.e. a plane.
* We can again draw two axes: one for each dimension. 
* If we use the same x-axis and y-axis to measure each point drawn on that paper, these two axes are said to define a coordinate system.
* If these two rulers are perpendicular to each other, they define what we call a Cartesian coordinate system.

* Note taht we commonly use a concise notation called an ordered pair to write the coordinates of a point.
* An ordered pair is simply two numbers separated by a comma. For Cartesian coordinate systems, it is customary to first write the horizontal x-coordinate followed by the vertical y-coordinate. 
* Remember, we can always interpret these ordered pairs as two signed distances.

* At this point in the lesson, we now know how to make a two-dimensional Cartesian coordinate system and define the coordinates of a 2D point in that coordinate system. 
* Note that the coordinates of points defined in a coordinate system are unique.
* This means that the same point cannot be represented by two different wsets of coordinates simultaneously in one system.
* However, it is important to note taht we are free to choose any coordinate system we please.

- Figure 2: a 2D Cartesian coordinate system is defined by two perpendicular (right angle) axes. Each axis is divided into regular intervals of unit length. Computing the coordinates of a 2D point is simply an extension of the 1D case to the 2D case. We take signed distances from the point to the origin of the coordinate system in both x and y.

* In fact, we can choose to define infinitely many such coordinate systems in a plane. For the sake of simplicity, let's assume that we drew just two such Cartesian coordinate systems on a sheet of paper. 
* On this paper we place one point.
* The coordinates of that point will be different depending on which of the two coordinate systems we consider.

* If you know the coordinates of *P* in coordinate system *A*, what do you need to do to find the coordinate of the same point in another coordinate system *B*?
* This represents an extremely important question in CG.
* We will soon learn why along with how to find the map which translates the coordinates of a poitn from one coordinate system to another.

-Figure 3: the same point is defined in two different coordinate systems. We can transform the point in the red coordinate system *A* to the green coordinate system *B* by adding the values (3,1) to its coordinates.

* For now, let's just consider the previous example in Figure 3. Note that by addint the values (3,1) coordinate-wise (that is, add the two x-axis values and then add the two y-axis values independently) to the coordinates (-1, 3) leads to the coordinate (2,4). So adding (3,1) to the coordinates fo *P* in *A* yields the coordinates of *P* in *B*. 
* Adding (-3,01) to (2,4) yields (-1, 3). This takes coordinates of *P* in *B* to coordinates of *P* in *A*.

* Another common operation is to move the point in the coordinate system *A* to another location in the same coordinate system.
* This is called a translation and is certainly one of the most basic operation you can do on points.
* Note that all sorts of other linear operators can be applied to coordinates.
* A multiplication of a real number to the coordinate of a point produces a scale (figure 4).
* A scale moves *P* along the line that is going through the poitn and the origin (because when we are transforming a point we are actually transforming the vector going from the origin to the point). 

- Figure 4: scaling or translating a point modifieds its coordinates. A scale is a mulitplication of the point's coordinates by some value. A translation is an addition of some values to the point's coordinates.

## The Third Dimension
* The 3D coordinate system is a simple extension of the 2D case. We will just be addint a third axis orthogonal to both the x- and y-axis called the z-axis (representative of depth). 
* The x-axis points to the right, the y-axis points up and the z-axis points backward.
* In Geometry, this 3D coordinate system defines what is more formally known as Euclidean space.

* In linear algebra, the three axes form what we call the basis of that coordinate system. 
* A basis is a set of linearly independent vectors that, in a linear combination, can represent every vector in a given vector space (the coordinate system).
* Vectors from a set are said to be linearly independent if and only if none of the vectors in the set can be written as a linear combination of other vectors in that set. 
* Change of basis, or change of coordinate system, is a common operation in mathematics and the graphics pipeline.

- Figure 5. a three dimensional coordinate system. A point is defined by three coordinates, one for each axis.

## Left-Handed vs Right-Handed Coordinate Systems 
Unfortunately, due to various conventions concerning handedness, coordinate systems are not that simple. 
* When the up and forward vectors are oriented in the same way (the forward vector is pointing away from the plane defined by the screen), and appropriate right vector can either point to the left or the right.

* To differentiate the two conventions, we call the first coordinate system the left-handed coordinate system, and the other, the right-handed coordinate system.

* Fittingly, your left hand orients the left-hand coordinate system while your right hand orients the right-hand coordinate system.

-Figure 6. typically the right-hand coordinate system is represented with the right axis pointing to the right, and the forward vector pointing away from the screen.

* Remember that the middle finger always represents the right vector.
* First orient the middle finger on either of your hands along what you consider to be the right vector and check if the other two fingers point in the same direction as the other two axes. From there , you shall see immediately if it is a left- or right-hand coordinate system. 

* The handedness of the coordinate system also plays a role in the orientation of normals computed from the edges of polygonal faces. 
* If the orientation is right-handed, then polygons whose vertices specified conterclockwise order will be front-facing. 

## The Right, Up and Forward Vectors
* The Cartesian coordinate system is only defined by three perpendicular vectors of unit length. 
* This coordinate system does not convey anything about what these three axes actually mean.
* The developer is the one that decides how these axes should be interpreted. 

- The most popular convention used in CG defines the up vector as being the y-axis. However, its is not uncommon to find in many CG related papers (particularly those related to shading techniques) coordinate systems where the up vector is defined as the z-axis. 

* The only thing that defines the handedness of the coordinate system is the orientation of the left (or right) vector relative to the up and forward vectors, regardless of what these axes represent.
* Handedness and conventions regarding the names of the axes are two different things. 

* It is also critically important to know which convention is used for the coordinate system when dealing with a renderer or any other 3D application. At present, the standard in the industry tends to be the right-hand XYZ coordinate system where x points to the right, y is up and z is outward. Programs and 3D APIs such as MAya and OpenGL use a right-hand coordinate system.
* Essentially, this means that the z-coordinate of 3 for a point in one system is -3 in the other. 
* For this reason, we potentially need to reverse the sign of an object's z-coordinate when the geometry is exported by the renderer. 

* It's actually easy to go from one coordinate system to another. All that is needed is to scale the point coordinates and the camera-to-world matrix by (1,1,-1).

## The World Coordinate System
* In most 3D applications, each different types of coordinate system is defined with respect to a master coordinate system called the world coordinate system.
* It defines the origin and the main x-,y- and z-axes from which all other coordinate systems are defined. 
* The world coordinate system is maybe the most important of all the distinct coordinate systems in the rendering pipeline. 
* These include the object, local (used in shading), camera and screen coordinate systems. 

# Math Operations on Points and Vectors
## Vector Length
* A vector can be seen as an arrow starting from one point and finishing to another. The vector itself indicates not only the direction of point *B* from *A* but also can be used to find out the distance between *A* and *B*. This is given by the length of a vector which can be easily computed with the following formula. The vector's length is sometimes also called norm or magnitude. 
* Note that the axes of the three-dimensional cartesian coordinate systems are unit vectors. 

## Normalizing a Vector
* A normalized vector is a vector whose length is 1. Such a vector is also called a unit vector (it is a vector which has unit length).
* Normalizing a vector is very simple. We first compute the length of the vector and divide each one of the vectors coordinates with this length.

* Note that the C++ implementation can be optimized. First we only normalize the vector if its length is greater than 0. We then compute a temporary variable which is the invert of the vector length, and multiply each coordinate of the vector with this value rather than dividing them with the vector's length.

* As you may know, multiplications in a program are less costly than divisions
- Figure 1. the magnitude or length of vector *A* and *B* is denoted by the double bar notation. A normalized vector is a vector whose length is 1.

## Dot Product
* The dot product or scalar product requires two vectors *A* and *B* can be seen as the projection of one vector onto the other. 
* The result of the dot product is a real number.
* The dot product consists of multiplying each element of the *A* vector with its counterpart from vector *B* and taking the sum of each product.
-Figure 2. the dot product of two vectors can be seen as the projection of *A* over *B*. If the two vectors *A* and *B* have unit length then the result of the dot product is the cosine of the angle subtended by the two vectors. \

* If we take the square root of the dot product between two vectors that are equal, then what we get is the length of the vector. 

* The dot product between two vectors is an extremely important and common operation in any 3D application because the result of this operation relates to the cosine of the angle between two vectors. 

* If *B* is a unit vector then A dot product B gives the cosine theta of A, the magnitude of the projection of *A* in the direction of *B*, with a minus sign if the direction is opposite. This is called the scalar projection of *A* onto *B*. 

* When the two vectors are normalized then taking the arc cosine of the dot product gives you the angle theta between the two vectors. 
* The dot product is a very important operation in 3D. It can be used for many things. As a test of orthogonality. When two vectors are perpendicular to each other, the result of the dot product betweeen these two vectors is 0.
* When the two vectors are pointing in opposite directions, the dot product returns 01. 
* When they are pointing in the exact same direction, it returns 1.
* It is also used intensively to find out the angle between two vectors or compute the angle between a vector and the axis of a coordinate system.

## Cross Product
* The cross product is also an operation on two vectors, but the difference is that the dot product returns a number, the cross product returns a vector.
* The particularity of this operation is that the vector resulting from the cross product is perpendicular to the other two. 
- Figure 3. the cross product of two vectors *A* and *B* gives a vector *C* perpendicualr to the plane defined by *A* and *B*. When *A* and *B* are orthogonal to each other (and have a unit length), *A*,*B*, *C* for ma Cartesian coordinate system.

* The result of the cross product is another vector which is orthogonal to the other two. 
* The two vectors *A* and *B* define a plane and the resulting vector *C* is perpendicular to that plane. 

* It is important to note that the order of vectors involved in the cross product has an effect on the resulting vector *C*. 

* In mathematics, the rseult of a cross product is called a pseudo vector. 

# Matrices
* Before we explain why matrices are interesting, let's start by saying that rendering an image by keeping all the 3D objects and the camera at the origin would be quite limited.
* In essence, matrices play an eesential roles in moving objects, light and cameras around in the scene so that you can compose your image the way you want.

## Introduction to Matrices: They Make Transformations Easy!
* In the previous chapter we mentioned that it was possible to translate or rotate points by using linear operators. For example, we showed that we could translate a poitn by adding some values to its coordinates. 
* We also showed that it was possible to rotate a vector by using trigonometric functions. 
* A matrix is just a way of combining all these transformations (scale, rotation, and translation) into one single structure.

* Multiplying a point or a vector by this structure gives us a transformed point or vector. 
* Combining these transformations means any combination of the following linear transformations: scale, rotation and translation.
* We can create a matrix that will rotate a point by 90 degrees around the x-axis, scale it by 2 along the z-axis and then translate it by (-2,3,1).
* Matrices can be used to combine together any of the three basic geometric transformations we can perform on points and vectors (scale, translation, and rotation) in a very easy, fast and compact way. 

## Point-Matrix Multiplication
* A point or a vector is a sequence of three numbers and for this reason they too can be written as a 1x3 matrix, a matrix that has one row and three columns.
* A point multiplied by a matrix transforms the point to a new position. The result of a point multiplied by a matrix has to be a point.

## The Scaling Matrix
* Note taht if either one of the scaling coefficients in the matrix are negativem, then the point's coordinate for the corresponding axis will be flipped (it will be mirrored to the other side of the axis).

## Orthogonal Matrices
* In fact the type of matrices we have just described in this chapter and the previous one (the rotation matrices), are called in linear algebra, orthogonal matrices. 
* An orthogonal matrix is a square amtrix with real entries whose columns and rows are orthogonal unit vectors. 
* We have mentioned previously that each row from the matrix represents an axis of a Cartesian coordinate system. 
* If the matrix is a rotation matrix or the result of several rotational matrices multiplied with each other, then each row necessarily represents an axis of unit length (because the elemnts of the rows are constructed from the sine and cosine trigonometric functions which are used to compute the coordinate of points lying on the unit circle).
* You can see them as a Cartesian coordinate system which is originally alinged with the world coordinate system (the identity matrix's rows represent the axes of the world coordinate system) and rotated around one particular axis or a random axis. 

* Orthogonal matrices have a few interesting properties, but the most useful one is that the transpose of an orthogonal matrix is equal to its inverse. 

## Affine Transformations
* You will sometimes find the terms affine transformations used in place of matrix transformation. This technical term is actually more accurate to designate the transformations that you get from using the type of matrices we have described so far. 
* An affine transformation, is a transformation that preserves straight lines.
* THe translation, rotation and shearing matrix are all affine transformations as are their combinations. 

* Each row of the matrix represents one axis of a Cartesian coordinate system. 
