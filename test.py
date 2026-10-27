import pystl
vec = pystl.vector(10)
vec[0] = 6
vec[1] = 7
vec[2] = 67

print("Vector size: ", len(vec))
print("First element: ",vec[0])
print("Size of vector: ", len(vec))
print("Max size of Vector: ", vec.maxsize())
vec.resize(6)
print("Size of vector: ", len(vec)) 
print("Max size of Vector: ", vec.maxsize())
print("Capacity of Vector: ", vec.capacity())
print("Vector is empty: ",vec.isEmpty())
vec.reserve(5)
vec.shrinkFit()
print("Max size of Vector: ", vec.maxsize())
print("Capacity of Vector: ", vec.capacity())
