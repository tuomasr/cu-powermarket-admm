#include <stdio.h>
#include <time.h>

#define E_PRI  10e-5
#define INF 9999999999

/*
Tuomas Rintamäki 2015
tuomas.rintamaki@aalto.fi
See Kraning et al. (2014) Dynamic Network Energy Management via Proximal Message Passing. Foundations and Trends in Optimization, 1(2):70-122.
http://stanford.edu/~boyd/papers/msg_pass_dyn.html
*/

void output_double_matrix(char folder[], char filename[], int m, int n, double *array) {
  FILE *f;
  int i,j;

  char path[100];
  strcpy(path,folder);
  strcat(path,filename);

  // open the output file
  f = fopen(path,"w");

  // output
  for (i=0;i<m;i++) {
    for (j=0;j<(n-1);j++) {
      fprintf(f,"%lf,",array[i*n+j]);
    }
    fprintf(f,"%lf",array[i*n+n-1]);
    fprintf(f,"\n");
  }

  fclose(f);
}

/*
Read problem dimensions from an external file
Each pointer variable holds the address of the actual variable
*/
void read_dim(char folder[], char filename[], int* T, int* nn, int* ng, int* nl, int* pieces, int* pars) {
  FILE *f;
  char path[100];
  strcpy(path,folder);
  strcat(path,filename);

  // open the file
  f = fopen(path,"r");

  // read dimensions
  fscanf(f,"%d",T);
  fscanf(f,"%d",nn);
  fscanf(f,"%d",ng);
  fscanf(f,"%d",nl);
  fscanf(f,"%d",pieces);
  fscanf(f,"%d",pars);
  
  // close the file
  fclose(f);
}


/*
Read load data from an external file
*/
void read_load(char folder[], char filename[], int T, int n, int nn, double* array) {
  int i;
  double val;
  FILE *f;
  char path[100];
  strcpy(path,folder);
  strcat(path,filename);

  // open file
  f = fopen(path,"r");

  // load data has structure time step x node
  // i*nn determines the row and n the column
  // read the values and make them negative
  for (i=0;i<T;i++) {
    fscanf(f,"%lf",&val);
    array[i*nn + n] = -val;    // make load figures negative
  }

  fclose(f);
}

/*
Read supply curve parameters
*/
void read_sc(char folder[], char filename[], int T, int g, int ng, int pieces, int pars, double* array) {
  int i,j,k;
  FILE *f;
  char path[100];
  strcpy(path,folder);
  strcat(path,filename);

  // open file
  f = fopen(path,"r");

  // supply curve data has the structure time step x generator x piece x parameters
  for (i=0;i<T;i++) {
    for (j=0;j<pieces;j++) {
      for (k=0;k<pars;k++) {
        fscanf(f,"%lf",&array[i*ng*pieces*pars + g*pieces*pars + j*pars + k]);
      }
    }
  }

  fclose(f);
}

/*
Read net transmission capacities (NTC)
*/
void read_ntc(char folder[], char filename[], int T, int l, int nl, double* array) {
  int i;
  FILE *f;
  char path[100];
  strcpy(path,folder);
  strcat(path,filename);

  // open file
  f = fopen(path,"r");

  // ntc data has the structure time step x 2*line (directions)
  for (i=0;i<T;i++) {
    fscanf(f,"%lf",&array[i*nl*2 + l*2]);
    fscanf(f,"%lf",&array[i*nl*2 + l*2 + 1]);
  }

  fclose(f);
}

/*
Read how nodes are connected to transmission lines
*/
void read_node_to_line(char folder[], char filename[], int t, int nn, int nl, int* array) {
  int i,j,offset;
  FILE *f;
  char path[100];
  strcpy(path,folder);
  strcat(path,filename);

  // open the file
  f = fopen(path,"r");

  offset = t*nl*2*nn;

  // data structure: 2*lines x nodes
  for (i=0;i<nl*2;i++) {
    for (j=0;j<nn;j++) {
      fscanf(f,"%d",&array[offset+i*nn+j]);
    }  
  }

  fclose(f);
}

/*
Read how transmission line flows are connected to nodes
*/
void read_line_to_node(char folder[], char filename[], int t, int nn, int nl, int* array) {
  int i,j,offset;
  FILE *f;
  char path[100];
  strcpy(path,folder);
  strcat(path,filename);

  // open the file
  f = fopen(path,"r");

  offset = t*nn*nl;

  // data structure: nodes x lines
  for (i=0;i<nn;i++) {
    for (j=0;j<nl;j++) {
      fscanf(f,"%d",&array[offset+i*nl+j]);
    }  
  }

  fclose(f);
}

/*
Read network sizes
*/
void read_net_size(char folder[], char filename[], int t, int nn, int* array) {
  int i,offset;
  FILE *f;
  char path[100];
  strcpy(path,folder);
  strcat(path,filename);

  // open the file
  f = fopen(path,"r");

  offset = t*nn;

  // read the number of terminals at each node
  for (i=0;i<nn;i++) {
    fscanf(f,"%d",&array[offset+i]);  
  }

  fclose(f);
}

/*
Compute node imbalance
*/
__global__ void KernelImbalance(int t, int nn, double *l, double *g, double *e, double *pbar_net, double *u_net, int *net_size, int *flag)
{
  int i = t*nn + threadIdx.x;
  pbar_net[i] = (l[i] + g[i] + e[i])/net_size[i];
  u_net[i] = u_net[i] + pbar_net[i];
  
  // check imbalance status
  if (pow(pbar_net[i],2.0) > E_PRI) {
    flag[i] = 1;
  } else {
    flag[i] = 0;
  }
}

/*
Check whether a node is balanced or not
*/
__device__ int flag_up(int t, int nn, int *flag) {
  int i, offset = t*nn;

  for (i=0;i<nn;i++) {
    if (flag[offset+i] == 1) {
      return 1;
    }
  }

  return 0;
}

/*
Map data from nodes to transmission lines
*/
__global__ void KernelMapNodeToLine(int t, int nn, int nl, double *pbar_net, double *u_net, double *pbar_net_line, double *u_net_line, int *node_to_line) {
  int i = t*nn + threadIdx.x;
  int j,k,offset1,offset2;

  offset1 = t*nl*2*nn;
  offset2 = t*nl*2;

  for (j=0;j<nl*2;j++) {
    k = offset1+j*nn+threadIdx.x;

    if (node_to_line[k]==1) {
      pbar_net_line[offset2+j] = pbar_net[i];
      u_net_line[offset2+j] = u_net[i];
    }
  }
}

/*
Compute the dual variable of a transmission flow
*/
__global__ void KernelLineDual(int t, int nn, int nl, double *v, double *flow, double *pbar_net_line, double *u_net_line)
{
  // each thread computes two elements in the v vector to avoid memory race conditions when accessing flows
  int i = t*nl*2 + threadIdx.x*2;
  int j = t*nl + threadIdx.x;

  v[i] = flow[j] - pbar_net_line[i] - u_net_line[i];
  v[i+1] = -flow[j] - pbar_net_line[i+1] - u_net_line[i+1];
}

/*
Find the optimal transmission flow (primal variable)
*/
__global__ void KernelOptimizeFlow(int t, int nl, double *ntc, double *flow, double *v) {  
  int i = t*nl + threadIdx.x;
  int j = t*nl*2 + threadIdx.x*2;

  // solve unconstrained case
  double uc = v[j] - 0.5*(v[j]+v[j+1]);

  // enforce bounds
  if (uc < -ntc[j]) {
		uc = -ntc[j];
	} else if (uc > ntc[j+1]) {
		uc = ntc[j+1];
  }

  flow[i] = uc;
}

/*
Map data from transmission lines to nodes
*/
__global__ void KernelMapLineToNode(int t, int nl, int nn, int *line_to_node, double *flow, double *exchange) {  
  int i = threadIdx.x;
  int j = t*nl + threadIdx.x;
  int k, offset;
  int l = t*nn + threadIdx.x;
  extern __shared__ double scaled[];

  offset = t*nn*nl;

  // multiply the incidence matrix with the flows
  for (k=0;k<nn;k++) {
      scaled[k*nl+i] = line_to_node[offset+k*nl+i]*flow[j];
  }

  // wait until all threads have finished the column computation
  __syncthreads();

  // compute row sums
  if (i<nn) {
    // reset the exchange figure
    exchange[l] = 0;

    for (k=0;k<nl;k++) {
      exchange[l] += scaled[i*nl+k];
    } 
  }
}

/*
Compute the dual variable of a generation variable
*/
__global__ void KernelGenDual(int t, int ng, int pieces, double *gen, double *v, double *pbar_net, double *u_net)
{
  int i = t*ng + threadIdx.x;
  int j = t*ng*pieces + threadIdx.x*pieces;
  int k;
  
  // build a T x ng x pieces matrix
  for (k=0;k<pieces;k++) {
    v[j+k] = gen[i] - pbar_net[i] - u_net[i];
  }
}

/*
Evaluate an quadratic function
*/
__device__ double quadratic(double a, double b, double c, double x, double v, double rho) {
	return a*pow(x,2.0)+b*x+c+(rho/2.0)*pow((x-v),2.0);
}

/*
An algorithm for computing the optimal value of a piecewise quadratic problem in parallel
*/
__global__ void KernelComputeGen(int t, int ng, int pieces, int pars, double *x1, double *x2, double *x3, double *y1, double *y2, double *y3, double *sc, double *v, double rho) {
  int i = t*ng*pieces*pars + threadIdx.x*pieces*pars + threadIdx.y*pars;
  int j = t*ng*pieces + threadIdx.x*pieces + threadIdx.y;

  // read parameters
  double a = sc[i];
  double b = sc[i+1];
  double c = sc[i+2];
  double lo = sc[i+3];
  double hi = sc[i+4];

  double vv = v[j];
  
  // compute points
  x1[j] = lo;
  x3[j] = hi;
  double mid = (rho*vv-b)/(2*a+rho);
  x2[j] = mid;

  // compute values
  y1[j] = quadratic(a,b,c,lo,vv,rho);
  y3[j] = quadratic(a,b,c,hi,vv,rho);
  
  if (lo<mid&&mid<hi) {
    y2[j] = quadratic(a,b,c,mid,vv,rho);
  } else {
    y2[j] = INF;
  }
}

/*
An algorithm for computing the optimal value of a piecewise quadratic problem in parallel
*/
__global__ void KernelOptimizeGen(int t, int ng, int pieces, double *x1, double *x2, double *x3, double *y1, double *y2, double *y3, double *gen) {
	int i,j,offset;
	double yy1, yy2, yy3;
	double min_val = INF, min_point = 0.0;
	
  offset = t*ng*pieces+threadIdx.x*pieces;

	// find the optimal generation level
	for (i=0;i<pieces;i++) {
		j = offset + i;
		yy1 = y1[j];
		yy2 = y2[j];
		yy3 = y3[j];
		
		if (yy1<yy2 && yy1<yy3 && yy1<min_val) {
			min_val = yy1;
			min_point = x1[j];
		} else if (yy2<yy1 && yy2<yy3 && yy2<min_val) {
			min_val = yy2;
			min_point = x2[j];
		} else if (yy3<yy1 && yy3<yy2 && yy3<min_val) {
			min_val = yy3;
			min_point = x3[j];
		}
	}
	
	// save the optimal generation
	gen[t*ng+threadIdx.x] = min_point;
}

/*
Update the step size parameter
*/
__global__ void KernelUpdateUnet(int t, int nn, double rr, double *u_net) {
  int i = t*nn + threadIdx.x;

  u_net[i] = rr*u_net[i];
}

/*
The kernel for ADMM - solves the optimization problem iteratively
*/
__global__ void KernelADMM(int nn, int ng, int nl, int pieces, int pars,
          double *l, double *sc, double *ntc, int *node_to_line, int *line_to_node, int *net_size,
		      double *x1, double *x2, double *x3, double *y1, double *y2, double *y3,
		      double *gen, double *flow, double *exchange, double *pbar_net, double *u_net,
		      double *v_gen, double *v_flow, double *pbar_net_line, double *u_net_line,  
          int *flag) {
  int t = blockIdx.x, iter = 1, max_iter = 10000;
  double rho = exp(-1.0/2.0)+0.03, rho0, rr;
  dim3 threadsPerBlockGen(ng,pieces);

  for (iter=1;iter<max_iter;iter++) {
    // compute imbalance at each node and update the imbalance flag
    KernelImbalance<<<1, nn>>>(t, nn, l, gen, exchange, pbar_net, u_net, net_size, flag);
    cudaDeviceSynchronize();
	
    // quit the algorithm if all flags are zero, i.e., the imbalance is less than required
    if (!flag_up(t,nn,flag)) {
      // scale the returned values
      KernelUpdateUnet<<<1, nn>>>(t, nn, -rho, u_net);
      printf("iterations %d\n",iter);
      return;
    }
    // otherwise, continue
	
    // optimize
    KernelGenDual<<<1, ng>>>(t, ng, pieces, gen, v_gen, pbar_net, u_net);
    KernelComputeGen<<<1, threadsPerBlockGen>>>(t, ng, pieces, pars, x1, x2, x3, y1, y2, y3, sc, v_gen, rho);
    KernelOptimizeGen<<<1, ng>>>(t, ng, pieces, x1, x2, x3, y1, y2, y3, gen);

    KernelMapNodeToLine<<<1, nn>>>(t, nn, nl, pbar_net, u_net, pbar_net_line, u_net_line, node_to_line);
    KernelLineDual<<<1, nl>>>(t, nn, nl, v_flow, flow, pbar_net_line, u_net_line);
    KernelOptimizeFlow<<<1, nl>>>(t, nl, ntc, flow, v_flow);
    KernelMapLineToNode<<<1, nl, nn*nl*sizeof(double)>>>(t, nl, nn, line_to_node, flow, exchange);
    
    // update rho
    rho0 = rho;
    rho = exp(-(double)iter/2)+0.03;
    rr = rho0/rho;
    // update u_net accordingly
    KernelUpdateUnet<<<1, nn>>>(t, nn, rr, u_net);
  }

  printf("The algorithm did not converge for timestep %d. \n", t);
  return;
}

int main()
{
  int i,t,T,nn,ng,nl,pieces,pars;
  char folder[] = "data/";
  char dim_file[] = "dim.txt";

  // &x indicates pointer to x, i.e. the address of x
  read_dim(folder,dim_file,&T,&nn,&ng,&nl,&pieces,&pars);
  printf("Time steps %d, nodes %d, generators %d, lines %d \n",T,nn,ng,nl);
  printf("Generator pieces %d and parameters %d \n",pieces,pars);

  // initialise host-side variables
  double *l, *sc, *ntc, *gen, *u_net;
  int *node_to_line, *line_to_node, *net_size;
  l = (double* )malloc(T*nn*sizeof(double));
  sc = (double* )malloc(T*ng*pieces*pars*sizeof(double));
  ntc = (double* )malloc(T*nl*2*sizeof(double));
  gen = (double* )malloc(T*ng*sizeof(double));
  u_net = (double* )malloc(T*nn*sizeof(double));

  node_to_line = (int *)malloc(T*nl*2*nn*sizeof(int));
  line_to_node = (int *)malloc(T*nn*nl*sizeof(int));
  net_size = (int *)malloc(T*nn*sizeof(int));
  
  char load_file[100], sc_file[100], ntc_file[100], net_size_file[100], node_to_line_file[100], line_to_node_file[100];

  // read load data
  for (i=0;i<nn;i++) {
    sprintf(load_file, "load%d.txt", i);
    read_load(folder,load_file,T,i,nn,l);
  }

  // read generator parameters
  for (i=0;i<ng;i++) {
    sprintf(sc_file, "gen%d.txt", i);
    read_sc(folder,sc_file,T,i,ng,pieces,pars,sc);
  }

  // read transmission line capacities (ntc's). There are two capacity figures per transmission line
  for (i=0;i<nl;i++) {
    sprintf(ntc_file, "line%d.txt", i);
    read_ntc(folder,ntc_file,T,i,nl,ntc);    
  }
  
  // read node to line matrix
  sprintf(node_to_line_file, "node_to_line.txt");
  for (t=0;t<T;t++) {
    read_node_to_line(folder,node_to_line_file,t,nn,nl,node_to_line);
  }

  // read line to node matrix
  sprintf(line_to_node_file, "line_to_node.txt");
  for (t=0;t<T;t++) {
    read_line_to_node(folder,line_to_node_file,t,nn,nl,line_to_node);
  }

  // read the number of terminals at each node
  sprintf(net_size_file,"net_size.txt");
  for (t=0;t<T;t++) {
    read_net_size(folder,net_size_file,t,nn,net_size);
  }
  
  // allocate memory on the device
  double *d_l, *d_sc, *d_ntc, *d_gen, *d_flow, *d_exchange, *d_pbar_net, *d_u_net, *d_v_gen, *d_v_flow, *d_pbar_net_line, *d_u_net_line;
  double *d_x1, *d_x2, *d_x3, *d_y1, *d_y2, *d_y3;
  int *d_node_to_line, *d_line_to_node, *d_net_size, *d_flag;
  
  // parameters
  cudaMalloc(&d_l, T*nn*sizeof(double));
  cudaMemcpy(d_l, l, T*nn*sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc(&d_sc, T*ng*pieces*pars*sizeof(double));
  cudaMemcpy(d_sc, sc, T*ng*pieces*pars*sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc(&d_ntc, T*nl*2*sizeof(double));
  cudaMemcpy(d_ntc, ntc, T*nl*2*sizeof(double), cudaMemcpyHostToDevice);
  
  // more parameters
  cudaMalloc(&d_node_to_line, T*nl*2*nn*sizeof(int));
  cudaMemcpy(d_node_to_line, node_to_line, T*nl*2*nn*sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc(&d_line_to_node, T*nn*nl*sizeof(int));
  cudaMemcpy(d_line_to_node, line_to_node, T*nn*nl*sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc(&d_net_size, T*nn*sizeof(int));
  cudaMemcpy(d_net_size, net_size, T*nn*sizeof(int), cudaMemcpyHostToDevice);
  
  // primal variables
  cudaMalloc(&d_gen, T*ng*sizeof(double));
  cudaMemset(d_gen, 0, T*ng*sizeof(double));
  cudaMalloc(&d_flow, T*nl*sizeof(double));
  cudaMemset(d_flow, 0, T*nl*sizeof(double));
  cudaMalloc(&d_exchange, T*nn*sizeof(double));
  cudaMemset(d_exchange, 0, T*nn*sizeof(double));
  
  // dual variables
  cudaMalloc(&d_pbar_net, T*nn*sizeof(double));
  cudaMemset(d_pbar_net, 0, T*nn*sizeof(double));
  cudaMalloc(&d_u_net, T*nn*sizeof(double));
  cudaMemset(d_u_net, 0, T*nn*sizeof(double));
  
  // temporary variables
  cudaMalloc(&d_v_gen, T*ng*pieces*sizeof(double));
  cudaMemset(d_v_gen, 0, T*ng*pieces*sizeof(double));
  cudaMalloc(&d_v_flow, T*nl*2*sizeof(double));
  cudaMemset(d_v_flow, 0, T*nl*2*sizeof(double));
  cudaMalloc(&d_pbar_net_line, T*nl*2*sizeof(double));
  cudaMemset(d_pbar_net_line, 0, T*nl*2*sizeof(double));
  cudaMalloc(&d_u_net_line, T*nl*2*sizeof(double));
  cudaMemset(d_u_net_line, 0, T*nl*2*sizeof(double));

  // more temporary variables
  cudaMalloc(&d_flag, T*nn*sizeof(int));
  cudaMemset(d_flag, 0, T*nn*sizeof(int));
  
  // even more temporary variables
  cudaMalloc(&d_x1, T*ng*pieces*sizeof(double));
  cudaMemset(d_x1, 0, T*ng*pieces*sizeof(double));
  cudaMalloc(&d_x2, T*ng*pieces*sizeof(double));
  cudaMemset(d_x2, 0, T*ng*pieces*sizeof(double));
  cudaMalloc(&d_x3, T*ng*pieces*sizeof(double));
  cudaMemset(d_x3, 0, T*ng*pieces*sizeof(double));
  cudaMalloc(&d_y1, T*ng*pieces*sizeof(double));
  cudaMemset(d_y1, 0, T*ng*pieces*sizeof(double));
  cudaMalloc(&d_y2, T*ng*pieces*sizeof(double));
  cudaMemset(d_y2, 0, T*ng*pieces*sizeof(double));
  cudaMalloc(&d_y3, T*ng*pieces*sizeof(double));
  cudaMemset(d_y3, 0, T*ng*pieces*sizeof(double));

  clock_t start = clock(), diff;

  KernelADMM<<<T,1>>>(nn, ng, nl, pieces, pars,
	d_l, d_sc, d_ntc, d_node_to_line, d_line_to_node, d_net_size,
	d_x1, d_x2, d_x3, d_y1, d_y2, d_y3,
	d_gen, d_flow, d_exchange, d_pbar_net, d_u_net,
	d_v_gen, d_v_flow, d_pbar_net_line, d_u_net_line,
	d_flag);
  cudaDeviceSynchronize();

  diff = clock() - start;
  int msec = diff * 1000 / CLOCKS_PER_SEC;
  printf("Time taken %d seconds %d milliseconds \n", msec/1000, msec%1000);
  
  // // copy results back to the host
  cudaMemcpy(u_net, d_u_net, T*nn*sizeof(double), cudaMemcpyDeviceToHost);

  char output_folder[] = "output/";
  char output_price[100];
  sprintf(output_price, "prices.csv");
  output_double_matrix(output_folder, output_price, T, nn, u_net);

  // free allocated memory
  // host
  free(l);
  free(sc);
  free(ntc);
  free(gen);
  free(u_net);
  
  free(node_to_line);
  free(line_to_node);
  free(net_size);

  // device
  cudaFree(d_l);
  cudaFree(d_sc);
  cudaFree(d_ntc);
  cudaFree(d_gen);
  cudaFree(d_flow);
  cudaFree(d_exchange);

  cudaFree(d_pbar_net);
  cudaFree(d_u_net);

  cudaFree(d_v_gen);
  cudaFree(d_v_flow);

  cudaFree(d_node_to_line);
  cudaFree(d_line_to_node);
  cudaFree(d_net_size);
  cudaFree(d_flag);
  
  cudaFree(d_x1);
  cudaFree(d_x2);
  cudaFree(d_x3);
  cudaFree(d_y1);
  cudaFree(d_y2);
  cudaFree(d_y3);

  return 0;
}