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
