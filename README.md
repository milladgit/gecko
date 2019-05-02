

# Gecko

This repo contains Gecko Runtime Library. With help of a set of directives, Gecko addresses multi-level memory hierarchy in current and future modern architectures and platforms.

![](./docs/images/hierarchy_tree.png)

------------------

## Constructs

###  Location

The principal constructs of *Gecko* model are *locations*. Locations are an abstraction of storage and computational resources available in the system and are represented as a node in *Gecko*’s tree-based hierarchy model. Similar to the Parallel Memory Hierarchy (PMH) model, *Gecko* is a tree of memory modules with workers at the leaves. Workers will perform computations and they attach to the memory modules. The figure shown below illustrates an example of our *Gecko* model. This model represents a regular cluster/supercomputer node with two non-uniform memory access (NUMA) multicore processors, *LocNi*, and four GPUs, *LocGi*, similar to ORNL’s Titan supercomputer.



### Location Type

The location hierarchy in *Gecko* designates how one location shares its allocated memories with another. When a memory is allocated in a location, the allocated memory is accessible by its children. They will have access to the same address as their parent has. However, the allocated memory is not shared with their parent and is considered to be a private memory with respect to their parent. The above figure shows how hierarchy will affect memory accesses among locations. The allocated memory *y* is allocated in Location *LocB* and consequently, it can be accessed by *LocB*, *LocN1*, and *LocN2*. However, *LocA* has no knowledge of the variable *y*. With the same logic, allocated memory *z* can be accessed by all locations, while accesses to *x* is only limited to *LocG1*.

In *Gecko*, locations are categorized by one of the followings: 1) a memory module, 2) a memory module with worker, and 3) a *virtual* location. Memory modules are annotated with their attributes (like type and size); *LocA* in the above figure is a memory module. If a location is a memory module with a worker attached to it, the location will be used to launch computational kernel by the runtime library; *LocNi* and *LocGi* are examples of memory modules with workers. Finally, the virtual locations, *LocB* and *LocC* in the above figure, are neither memory modules nor computational ones. They are an abstract representation of their children in the hierarchy. Grouping a set of locations under a virtual location provides a powerful feature for the application to address the location set as a whole. Similar approaches have been observed in related work. Like other location categories, virtual locations can also be the target of memory allocation and kernel launch requests. Depending on the type of requests and hints from scientific developers, the runtime library will act upon the requests and perform them at the execution time.

Locations are abstractions of available resources in the system. Any location in *Gecko*, internal or leaf locations, is possibly the target of a kernel launch by application. Virtual locations, however, provide flexibility to the applications. With virtual locations, applications aptly fix the target of their kernel to that location while changing the underlying structure of the tree for the same location. As a result, the application targeted for a multicore architecture dynamically or statically morphs into a program targeting different types of accelerators (e.g., NVIDIA GPUs, AMD GPUs, or FPGAs).

For further details, please refer to the Gecko's paper.


## Directives

Below are the set of Gecko's directives.

**Hint:** All keywords are mandatory unless they are put inside `[]`, which in that case, they are considered as optional.

### Location Type
The first step in utilizing Gecko is to declare all location types. This is achieved by the `loctype` construct. It accepts the following keywords:

```C++
#pragma gecko loctype name(char*) kind(char*, char*)
```

 - `name`: the name of the location type (`char*`).
 - `kind`: the type of the location type (`char* , char*`) from the declared ones by `loctype`.


**Note:** The `virtual` name is reserved for the virtual location type.

**Note 2:** This construct is a Work-In-Progress (WIP) and experimental. At this stage, we support extra keywords that are shown in the example below.

**Note 3:** For a list of supported memory `kind`s, please visit *the following example*.

**Example:** The first line in the following code snippet declares that our application will require a host CPU with 4 cores and of Intel's Skylake type. The second line declares a minimum NVidia GPU with Compute Capability of 5.0 (`cc50`) and names it `tesla` for future references in defining locations. The third line specifies the main memory module in our system with 16 GB in size. 


```C++
#pragma gecko loctype name("host") kind("x64", "Skylake") num_cores(4) mem("4MB")
#pragma gecko loctype name("tesla") kind("NVIDIA", "cc50") mem("4GB")
#pragma gecko loctype name("NODE_MEMORY") kind("Unified_Memory") size("16GB")
#pragma gecko loctype name("HDD") kind("Permanent_Storage") 
```



### Location

Locations in Gecko are defined using the `location` construct as shown below. It accept the following keywords:

```C++
#pragma gecko location name(char*) type(char*) [all]
```

 - `name`: the name of the location (`char*`).
 - `type`: the type of the location (`char*`) from the declared ones by `loctype`.
 - `all`: defining all devices with similar location type under one umbrella. This will result in the following naming convetion: `<name>[i]` where `<name>` is the name of the location and `[i]` provides a way to distinguish the locations.




**Example:** below lines define the locations used in the model shown above.

```C++
#pragma gecko location name("LocA") type("NODE_MEMORY")
#pragma gecko location name("LocB","LocC") type("virtual")
#pragma gecko location name("LocN1", "LocN2") type("host")
#pragma gecko location name("LocG1", "LocG2", "LocG3", "LocG4") type("tesla")
```

### Hierarchy

The hierarchy in Gecko determines the relationship among locations with respect to each other. Every hierarchy in Gecko has two associated entities: a parent and a child. Both of them are predefined locations.

```C++
#pragma gecko hierarchy children(<op> : <list>) parent(char*) [all]
```

 - `parent`: the parent location in the relationship (`char*`).
 - `children`: the list of all locations to be the children of `parent`. The keyword accepts an operation and the children list: `(<op> : <list>)`. The `op` can be `+` and `-` signs or a `char` variable that is either `+` or `-`. The `<list>` is the comma-separated list of defined locations (name of the locations in `char*`).
 - `all`: similar to the `all` keyword in the `hierarchy`, this keyword is used to include `all` locations under this hierarchy.




**Example:** Below lines shows how to use hierarchy construct. *Please pay attention to the three approach that we used the operation in our examples. All three perform exactly the same operation.*

```C++
char op = '+';
#pragma gecko hierarchy children(+:"LocB","LocC") parent("LocA")
#pragma gecko hierarchy children('+':"LocN1","LocN2") parent("LocB")
#pragma gecko hierarchy children(op:"LocG1","LocG2","LocG3","LocG4") parent("LocC")
```

**Note:** at this point, we have constructed the whole tree in the example shown in the above figure.



## Configuration File

The whole structure of the hierarchy tree can be stored within a configuration file. Gecko can load such file and populate the tree. This brings a great flexibility to the user and makes the application extremely portable. Its keywords are shown below:

 - `file`: the configuration file name.
 - `env`: the `GECKO_CONFIG_FILE` environment variable contains the path to the file. Please refer to the section describing the environmental variables below.

**Note:** The `file` and `env` could not be chosen simultaneously. 

```C++
#pragma gecko config env
#pragma gecko config file("/path/to/config/file")
```


**Example:** An example of a configuration file for above-mentioned hierarchy tree is shown below:

```CSV
loctype;kind,x64,Skylake;num_cores,4;mem,4MB;name,host;
loctype;name,tesla;kind,CC7.0,Volta;mem,4GB
loctype;name,NODE_MEMORY;kind,Unified_Memory;size,16GB

location;name,LocA;type,NODE_MEMORY;
location;name,LocB,LocC;type,virtual
location;name,LocN1,LocN2;type,host;
location;name,LocG1,LocG2,LocG3,LocG4;type,tesla

hierarchy;children,+,LocB,LocC;parent,LocA
hierarchy;children,+,LocN1,LocN2;parent,LocB
hierarchy;children,+,LocG1,LocG2,LocG3,LocG4;parent,LocC
```


## Drawing

For convenience, Gecko can generate the hierarchical tree for visualization purposes. Using `draw` construct, at any point at executing the program, the runtime library will generate a compatible DOT file. Eventually, one can convert a DOT file to a PDF file using the dot command: `dot -Tpdf gecko.conf -o gecko-tree.pdf`

The syntax to use `draw` is as following. It accepts a `root` keyword, which user should provide the location name of the root node in the hierarchy.

```C++
#pragma gecko draw root(char*) filename(char*)
```

 - `root`: the root node in the hierarchy tree (`char*`).
 - `filename`: the target DOT file name (`char*`). It can be absolute or relative path. The default value for this keyword is `"gecko.dot"`.



**Example:**

```C++
#pragma gecko draw root("LocA")
```



## Memory Operations

### Allocating/Freeing memory

Memory operations in Gecko are supported by the `memory` construct. To allocate memory, use `allocate` keyword and to free the object, use `free`. Optional features are specified inside brackets (`[]`).

```C++
#pragma gecko memory allocate(<ptr>[0:<count>]) type(<datatype>) location(char*) [distance(<dist>) [realloc/auto]] [file(char*)] 
#pragma gecko free(<ptr_list>)
```

 - `allocate(<ptr>[0:<count>])`: the input to the `allocate` keyword accepts a pointer (`<ptr>`) and its number of elements (`<count>`). `<count>` can be a constant or a variable. *Please see the example below.*
 - `datatype`: the data type of the `ptr` variable.
 - `<ptr_list>`: list of allocated variables by `allocate`. It can have one or more than one variable.
 - `<dist>`: specifies the distance of the allocation in distance-based allocations. For these types of allocations, the allocation is performed when the destination location to execute the region is chosen. As a result, the allocation is postponed until the region is ready to be executed. It accepts following values:
	 - `near`: the allocation is performed in the chosen execution location.
	 - `far[:<n>]`: the allocation is performed in the `n`-th grandparent with respect to the chosen execution location. For `n==0` and `n==1`, the immediate grandparent is chosen. In cases that `n` causes the location to be chosen to go further than the root location, the root location is chosen.
 - `<realloc/auto>`: the policy to perform the allocation. 
	 - `realloc`: the allocated memory is freed after the associated region is finished.
	 - `auto`: the allocated memory is not freed after the region is finished and it is moved around the hierarchy as needed. The allocated memory is able to be generalized (goes up in the hierarchy) and not privatized (going down the hierarchy). This feature is a **Working-In-Progress**.
 - `file`: the file name in case the location type is `Permanent_Storage` .


**Example**:

```C++
int N = 2000;
// place-holders for our memory locations
double *X, *Y, *Z, *W;
#pragma gecko memory allocate(X[0:N]) type(double) location("LocA")
#pragma gecko memory allocate(Y[0:N]) type(double) location("LocB")
#pragma gecko memory allocate(Z[0:N]) type(double) location("LocC")
#pragma gecko memory allocate(W[0:N]) type(double) location("LocG1")
//...<some computation>...
#pragma gecko memory free(X, Y, Z, W)
```

**Note:** Please refer to the `region` section to see an example of distance-based allocations.



### Copy and Move

One can copy memory between two different memory locations. It is similar to `memcpy` operations, however, it is performed between different platforms and architectures. 

```C++
// Copying elements from FVar[s1, e1] to TVar[s2, e2]
#pragma gecko memory copy from(FVar[s1:e1]) to(TVar[s2:e2])
```


**Example**:

```C++
#pragma gecko memory copy from(X[0:N]) to(Y[0:N])
```

In some cases, we have to move a set of data elements from Location P to Location Q. In such cases, the source location no longer possesses the variable and the destination locations has to own the variable.  It is like the variable was originally allocated in the destination location.

```C++
// Copying elements from FVar[s1, e1] to TVar[s2, e2]
#pragma gecko memory move(<var>) to(char*)
```


**Example**:

```C++
#pragma gecko memory move(Q) to("LocA")
```


### Register/Unregister

There are many cases where we are dealing with variables that are already allocated with a well-known allocation API and we want to use such variables in our location. However, they are unknown to Gecko. With `register/unregister` clauses one can introduce them properly to Gecko.


```C++
#pragma gecko memory register(<var>[<start>:<end>]) type(<type>) loc(char*)
#pragma gecko memory unregister(<var>) 
```

 - `<var>`: the already allocated memory.
 - `<start>` and `<end>`: start and end indices of the `var`.
 - `<type>`: type of the memory.
 - `loc`: the name of the proper location that this variable was originated from.


**Example**:

```C++
vector<double> v(100);
double *v_addr = (double*) v.data();
#pragma gecko memory register(v_addr[0:100]) type(double) loc("LocN")
double *d_addr;
cudaMalloc((void**) &d_addr, sizeof(double) * 100);
#pragma gecko memory register(d_addr[0:100]) type(double) loc("LocG1")
// ...
#pragma gecko memory unregister(d_addr) 
#pragma gecko memory unregister(v_addr) 
```




## Region

Gecko recognizes the computational kernels with the `region` construct. The end of the region is recognized with the `end` keyword.

```C++
#pragma gecko region exec_pol(char*) variable_list(<ptr_list>) [gang[(<gang_count>)]] [vector[(<vector_count>)]] [independent] [reduction(<op>:<var>)] [at(char*)] 
// the for loop
#pragma gecko region end
```

 - `datatype`: the execution policy to execute the kernel. *Please refer to execution policy section for more details.*
 - `<ptr_list>`: list of utilized variables within the region.
 - `gang`, `vector`, `independent`, and `reduction`: please refer to the OpenACC specification to learn more about these keywords.
	 - **Note:** Gecko relies on OpenACC to generate code for different architecture.
 - `at`: *[optional]* the destination location to execute the code. 
	 - **Note:** In the new version of Gecko, the destination location is chosen based on the variables used in the region (`<ptr_list>`). However, the developer can override Gecko and specify where to execute the code.


**Example:**
```C++
double coeff = 3.4;
int a = 0;
int b = N;
#pragma gecko region exec_pol("static") variable_list(Z,X)
for (int i = a; i<b; i++) {
	Z[i] = X[i] * coeff;
}
#pragma gecko region end
```


**Example of distance-based allocations:**

Following is an example of distance-based allocations.

```C++
double *T1, *T1_realloc, *T1_auto, *T2, *T2_far2, *T2_far_variable;
#pragma gecko memory allocate(T1[0:N]) type(double) distance(near)
#pragma gecko memory allocate(T1_realloc[0:N]) type(double) distance(near) realloc
#pragma gecko memory allocate(T2[0:N]) type(double) distance(far) file("T2.obj")
#pragma gecko memory allocate(T2_far2[0:N]) type(double) distance(far:2) file("T2_far.obj")
int far_distance = 10;
#pragma gecko memory allocate(T2_far_variable[0:N]) type(double) distance(far:far_distance) file("T2_far_variable.obj")

double *Perm;
#pragma gecko memory allocate(Perm[0:N]) type(double) location("LocHDD") file("perm.obj")

a = 0;
b = N;
long total = 0;
#pragma gecko region exec_pol("static") variable_list(Perm,Z,X,T1,T2_far_variable) reduction(+:total)
for (int i = a; i<b; i++) {
	Z[i] = X[i] * coeff;
	T1[i] *= 2;
	T2_far_variable[i] *= 2;
	total += (i+1);
	Perm[i] = i;
}
#pragma gecko region end
```



## Synchronization point (wait)

By default, regions in Gecko are executed asynchronously. Synchronization points in Gecko are expressed with the `pause` construct. The granularity at which the synchronization happens can be controlled with the `at` keyword. The location is an optional input. If the location is not specified, Gecko waits for all resources to finish their work.

 - `at`: the location to wait on

```C++
#pragma gecko region pause [at(char*)]
```

**Example:**
```C++
#pragma gecko region pause at("LocA")
```

------------------


## Execution Policies

Gecko supports a number of execution policies by default. Among those, we have `static`, `flatten`, `range`, `percentage`, and `any`.

In static distribution, the iteration space is divided evenly among children of that location. This policy is similar to the ones with the same name in OpenACC and OpenMP. Suppose we have a for-loop with 1,000,000 iterations. For our above-mentioned example tree, if we target location LocC, Gecko will assign each LocGs[i] location 250,000 iterations to process.

Flatten distribution is similar to the static distribution in its even distribution approach. In static, all the siblings of a location contribute equally in the distribution of workload. However, in flatten, all leaves (workers) contribute equally. Let's suppose LocB becomes a child of LocB and 1,000,000 iterations are about to be launched on LocB. Static strategy implies that each child of LocB is assigned 500,000 iterations. Then, Gecko splits the iteration space for LocB and LocC. As a result, each child of LocB operates on 250,000 iterations while each child of LocC operates on 125,000 iterations. However, flatten strategy divides the iteration space equally among leaf locations. Every child in both LocB and LocC locations will operate on 166,666 iterations. So, when flatten policy is chosen, the runtime finds the number of available workers below the chosen location in the tree and splits the iteration space evenly.

Gecko provides more flexibility in workload distribution with customized iteration ranges. An application is able to fine-tune the assigned ranges to each location hoping that such distribution leads to a better workload balance among workers in those locations. In such cases, one can provide an array of percentages or numbers to represent the policy. Considering the previous example with 1,000,000 iterations, one can provide percentage[20, 30, 40, 10] or range[200000, 300000, 400000, 100000] as a distribution policy to Gecko, which both approaches lead to the same workload strategy. In case the targeted location, similar to LocB, does not have four children, Gecko will assign the percentage and numbers from the beginning to children of the targeted location. The rest of the numbers in the list are assigned to other locations in a round-robin or work-stealing fashion. The static strategy is a special case of percentage and range strategies where we have percentage[25, 25, 25, 25] or range[250000, 250000, 250000, 250000] in our example.

In some cases, we are only interested in engaging only one of the locations of the hierarchy. As long as only a single location is chosen, an application is not committed to run in a specific location. In such cases, Gecko will find an idle location among children of the chosen target. Alternatively, based on the recorded history, Gecko can also choose the best architecture for this kernel if we are targeting a multi-architecture virtual location. For the above-mentioned example, in our current implementation, all the 1,000,000 iterations are assigned to a child of LocC in the hierarchy, if LocC is chosen as the target.


------------------

## Environmental Variables

### GECKO_HOME

 - Compile-time variable.
 - Specifies the path to the Gecko's folder.

### GECKO_CONFIG_FILE

 - Run-time variable.
 - Specifies the path to a configuration file. If this variable is set, the file that this path is pointing to will be used to populate the hierarchy tree.

### GECKO_POLICY

 - Run-time variable.
 - Specifies the execution policy for all the regions in the code. If it is set, all the regions in the code will be executed with a policy among the described one above.


------------------

## How to use Gecko?

#### Manual compilation
Defining following aliases will help the development.

```bash
alias gf="python /path/to/gecko/geckoTranslate.py forward ./"
alias gb="python /path/to/gecko/geckoTranslate.py backward ./"
```

The first alias, `gf`, if called within any folder, converts all the files annoted with the Gecko directives. It will replace the file with translated content. The translated file is ready to be compiled.

The second alias, `gb`, will revert back the Gecko-annotated files to their original version.

By calling `make` in the Gecko folder, the Gecko runtime library will be created in the `./lib` folder.

**How to compile the generated code?**
```bash
pgc++ -m64 -std=c++11 -w -Mllvm  -mp -acc -ta=tesla -Minfo=accel -DCUDA_ENABLED -I/usr/local/cuda/include/ -O3  -o <output_exe_file>  <source_file> -lm -L/usr/local/cuda/lib64 -lcudart  ./lib/libgecko.a
```

**Note:** Please refer to the Makefile and `test.cpp` files for examples of how to use Gecko.

#### Automatic compilation
For sake of simplicity, Gecko provides a Python script to compile/build the Gecko-annotated source codes easily. 

Define following alias will also help the development:

```bash
alias gm="python /path/to/gecko/geckoMake.py"
```

Then, we can compile our source code with `gm`. It performs `gf`, builds all the files, and then performs `gb` command to revert the files back to their original version.

If the package was already build with `make all` command, then we can build our package as following:
```bash
gm make all
```

**Note:** Please note to add the `libgecko.a` to your Makefile for the link command.

------------------

## Important Disclaimer

 - The `geckoTranslate.py` script is not a sophisticated compiler by any means. We tried our best to translate the directives of Gecko and generate a correct C++ code. There might be some cases that is not covered in our translation process.
 - The memory allocations are relying on the CUDA runtime library and `malloc`. 
 - The final code generated by `geckoTranslate.py` script contains OpenMP and OpenACC directives to target CPU (host) and GPU devices. The computational regions of the code rely on these directive-based programming models to utilize available hardware. 

------------------

## Citing Gecko

To cite Gecko, please use following.

#### ACM Format:
Millad Ghane, Sunita Chandrasekaran, and Margaret S. Cheung. 2019. Gecko: Hierarchical Distributed View of Heterogeneous Shared Memory Architectures. In _Proceedings of the 10th International Workshop on Programming Models and Applications for Multicores and Manycores_(PMAM'19), Quan Chen, Zhiyi Huang, and Min Si (Eds.). ACM, New York, NY, USA, 21-30. DOI: https://doi.org/10.1145/3303084.3309489

#### Bib Entry:
```
@inproceedings{ghane2019gecko,
	author = {Ghane, Millad and Chandrasekaran, Sunita and Cheung, Margaret S.},
	title = {Gecko: Hierarchical Distributed View of Heterogeneous Shared Memory Architectures},
	booktitle = {Proceedings of the 10th International Workshop on Programming Models and Applications for Multicores and Manycores},
	series = {PMAM'19},
	year = {2019},
	isbn = {978-1-4503-6290-0},
	location = {Washington, DC, USA},
	pages = {21--30},
	numpages = {10},
	url = {http://doi.acm.org/10.1145/3303084.3309489},
	doi = {10.1145/3303084.3309489},
	acmid = {3309489},
	publisher = {ACM},
	address = {New York, NY, USA},
	keywords = {Abstraction, Heterogeneous, Hierarchy, Portable, Programming Model, Shared Memory},
}
```

------------------
## Funding

This project is sponsored by generous support from the National Science Foundation (Award No. 1531814) and the Department of Energy (Award No. DE-SC0016501).

![](./docs/images/NSF_4-Color_bitmap_Logo_thumb.png)
![](./docs/images/DOE_logo_thumb.png)


------------------

## Contact

For any questions, please reach out to Millad Ghane (mghane2@uh.edu).
