/***********************************************************************/
/*                                                                     */
/*   svm_struct_api.c                                                  */
/*                                                                     */
/*   Definition of API for attaching implementing SVM learning of      */
/*   structures (e.g. parsing, multi-label classification, HMM)        */ 
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 03.07.04                                                    */
/*                                                                     */
/*   Copyright (c) 2004  Thorsten Joachims - All rights reserved       */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/***********************************************************************/

#include <stdio.h>
#include <string.h>
#include "svm_struct_common.h"
#include "svm_struct_api.h"
#include <vector>
#include <bitset>
#include <algorithm>

using std::vector;
using std::bitset;

#define MAX(x,y)      ((x) < (y) ? (y) : (x))
#define MIN(x,y)      ((x) > (y) ? (y) : (x))

void        svm_struct_learn_api_init(int argc, char* argv[])
{
  /* Called in learning part before anything else is done to allow
     any initializations that might be necessary. */
}

void        svm_struct_learn_api_exit()
{
  /* Called in learning part at the very end to allow any clean-up
     that might be necessary. */
}

void        svm_struct_classify_api_init(int argc, char* argv[])
{
  /* Called in prediction part before anything else is done to allow
     any initializations that might be necessary. */
}

void        svm_struct_classify_api_exit()
{
  /* Called in prediction part at the very end to allow any clean-up
     that might be necessary. */
}

SAMPLE      read_struct_examples(char *file, STRUCT_LEARN_PARM *sparm) {
    /* Reads struct examples and returns them in sample. The number of
       examples must be written into sample.n */
    SAMPLE sample;  /* sample */
    EXAMPLE *examples;
    long n;       /* number of examples */

    /* fill in your code here */

    long line_num, max_words_doc, max_line_length;
    char *line;
    nol_ll(file, &line_num, &max_words_doc, &max_line_length);
    n = line_num;

    examples = (EXAMPLE *) malloc(sizeof(EXAMPLE) * n);

    max_line_length += 10;
    line = new char[max_line_length];

    FILE *datafile;
    if ((datafile = fopen(file, "r")) == NULL) {
        printf("open error\n");
        exit(1);
    }
    printf("reading line...");
    fflush(stdout);

    int num = 0;
    while ((!feof(datafile)) && fgets(line, (int) max_line_length, datafile)) {
        PATTERN x;
        LABEL y;
        read_instance_struct(line, x, y);
        examples[num].x.pword = x.pword;
        examples[num].y.y = y.y;
        num++;
    }

    sample.n = num;
    sample.examples = examples;
    return (sample);
}

void  read_instance_struct(char *line, PATTERN &x, LABEL &y) {
    long pos = 0;
    y.y = new int[2];
    y.y[0]=read_num<int>(line, pos);
    y.y[1]=read_num<int>(line, pos);

    PWORD *word = new PWORD[1];
    word->col_num = read_num<int>(line, pos);
    word->row_num = read_num<int>(line, pos);

    long total = 0;
    for(int i = 0; i<word->row_num; i++){
        vector<float> val;
        for(int j = 0; j<word->col_num; j++){
            val.push_back(read_num<float>(line,pos));
            total++;
        }
        word->val.push_back(val);
    }
    if(total != word->col_num * word->row_num){
        printf("the number of pixes of the picture is wrong");
        fflush(stdout);
        exit(-1);
    }
    x.pword = word;
}


void        init_struct_model(SAMPLE sample, STRUCTMODEL *sm, 
			      STRUCT_LEARN_PARM *sparm, LEARN_PARM *lparm, 
			      KERNEL_PARM *kparm) {
    /* Initialize structmodel sm. The weight vector w does not need to be
       initialized, but you need to provide the maximum size of the
       feature space in sizePsi. This is the maximum number of different
       weights that can be learned. Later, the weight vector w will
       contain the learned weights for the model. */

    sm->num_multilabel = 0;
    int bw = 100000;
    int bh = 100000;
    for (int i = 0; i < sample.n; i++) {
        PWORD *pword = sample.examples[i].x.pword;
        if (bw > pword->col_num) bw = pword->col_num;
        if (bh > pword->row_num) bh = pword->row_num;
    }
    sm->bw = bw / 2;
    sm->bh = bh / 2;
    sm->sizePsi = sm->bw * sm->bh;
    sparm->bw = sm->bw;
    sparm->bh = sm->bh;
    sm->w = (double *) malloc(sizeof(double) * sm->sizePsi);
    memset(sm->w, 0, sizeof(double) * sm->sizePsi);
}

CONSTSET    init_struct_constraints(SAMPLE sample, STRUCTMODEL *sm, 
				    STRUCT_LEARN_PARM *sparm) {
    /* Initializes the optimization problem. Typically, you do not need
       to change this function, since you want to start with an empty
       set of constraints. However, if for example you have constraints
       that certain weights need to be positive, you might put that in
       here. The constraints are represented as lhs[i]*w >= rhs[i]. lhs
       is an array of feature vectors, rhs is an array of doubles. m is
       the number of constraints. The function returns the initial
       set of constraints. */
    CONSTSET c;
    c.lhs = NULL;
    c.rhs = NULL;
    c.m = 0;
    return (c);
}

LABEL       classify_struct_example(PATTERN x, STRUCTMODEL *model,
				    STRUCT_LEARN_PARM *sparm) {
    /* Finds the label yhat for pattern x that scores the highest
       according to the linear evaluation function in sm, especially the
       weights sm.w. The returned label is taken as the prediction of sm
       for the pattern x. The weights correspond to the features defined
       by psi() and range from index 1 to index sm->sizePsi. If the
       function cannot find a label, it shall return an empty label as
       recognized by the function empty_label(y). */
    LABEL y;

    int first = 1;
    vector<int> bestbox;
    bestbox.push_back(1);
    bestbox.push_back(1);
    double bestscore = -1000;
    int w = x.pword->col_num;
    int h = x.pword->row_num;
    int bw = model->bw;
    int bh = model->bh;

    double score;
    int *bbx = new int[2];
    for(int hi = 1; hi + bh - 1 <= h; hi+= 1){
        for(int wi = 1; wi + bw - 1 <= w; wi+=1){
            bbx[0]=wi;
            bbx[1]=hi;
            DOC doc;
            y.y=bbx;
            doc.fvec = psi(x, y,model,sparm);
            score = classify_example(model->svm_model, &doc);
            free_svector(doc.fvec);
            if((bestscore < score) || first){
                bestscore = score;
                bestbox[0] = bbx[0];
                bestbox[1] = bbx[1];
                first = 0;
            }
        }
    }
    y.y = new int[2];
    y.y[0] = bestbox[0];
    y.y[1] = bestbox[1];
    delete[] bbx;
    return y;
}


LABEL       find_most_violated_constraint_slackrescaling(PATTERN x, LABEL y, 
						     STRUCTMODEL *sm,
						     STRUCT_LEARN_PARM *sparm)
{
  /* Finds the label ybar for pattern x that that is responsible for
     the most violated constraint for the slack rescaling
     formulation. For linear slack variables, this is that label ybar
     that maximizes

            argmax_{ybar} loss(y,ybar)*(1-psi(x,y)+psi(x,ybar))

     Note that ybar may be equal to y (i.e. the max is 0), which is
     different from the algorithms described in
     [Tschantaridis/05]. Note that this argmax has to take into
     account the scoring function in sm, especially the weights sm.w,
     as well as the loss function, and whether linear or quadratic
     slacks are used. The weights in sm.w correspond to the features
     defined by psi() and range from index 1 to index
     sm->sizePsi. Most simple is the case of the zero/one loss
     function. For the zero/one loss, this function should return the
     highest scoring label ybar (which may be equal to the correct
     label y), or the second highest scoring label ybar, if
     Psi(x,ybar)>Psi(x,y)-1. If the function cannot find a label, it
     shall return an empty label as recognized by the function
     empty_label(y). */
  LABEL ybar;



  return(ybar);
}

LABEL       find_most_violated_constraint_marginrescaling(PATTERN x, LABEL y, 
						     STRUCTMODEL *model,
						     STRUCT_LEARN_PARM *sparm) {
    /* Finds the label ybar for pattern x that that is responsible for
       the most violated constraint for the margin rescaling
       formulation. For linear slack variables, this is that label ybar
       that maximizes

              argmax_{ybar} loss(y,ybar)+psi(x,ybar)

       Note that ybar may be equal to y (i.e. the max is 0), which is
       different from the algorithms described in
       [Tschantaridis/05]. Note that this argmax has to take into
       account the scoring function in sm, especially the weights sm.w,
       as well as the loss function, and whether linear or quadratic
       slacks are used. The weights in sm.w correspond to the features
       defined by psi() and range from index 1 to index
       sm->sizePsi. Most simple is the case of the zero/one loss
       function. For the zero/one loss, this function should return the
       highest scoring label ybar (which may be equal to the correct
       label y), or the second highest scoring label ybar, if
       Psi(x,ybar)>Psi(x,y)-1. If the function cannot find a label, it
       shall return an empty label as recognized by the function
       empty_label(y). */
    LABEL ybar;

    int w = x.pword->col_num;
    int h = x.pword->row_num;
    int bw = model->bw;
    int bh = model->bh;

    int first = 1;
    vector<int> bestbox;
    bestbox.push_back(1);
    bestbox.push_back(1);
    double bestscore = -1000;

    double score = 0;
    int *bbx = new int[2];
    for(int hi = 1; hi + bh-1 <= h; hi+= 10){
        for(int wi = 1; wi + bw-1 <= w; wi+=10){
            bbx[0]=wi;
            bbx[1]=hi;
            DOC doc;
            ybar.y=bbx;
            doc.fvec = psi(x, ybar, model,sparm);
            score = classify_example(model->svm_model, &doc);
            free_svector(doc.fvec);
            score += loss(y,ybar,sparm);
            if((bestscore < score) || first){
                bestscore = score;
                bestbox[0] = bbx[0];
                bestbox[1] = bbx[1];
                first = 0;
            }
        }
    }
    ybar.y = new int[2];
    ybar.y[0] = bestbox[0];
    ybar.y[1] = bestbox[1];
    delete[] bbx;
    return (ybar);
}

int         empty_label(LABEL y)
{
  /* Returns true, if y is an empty label. An empty label might be
     returned by find_most_violated_constraint_???(x, y, sm) if there
     is no incorrect label that can be found for x, or if it is unable
     to label x at all */

  return(y.y[0] == -1);
}

SVECTOR     *psi(PATTERN x, LABEL L, STRUCTMODEL *model,
		 STRUCT_LEARN_PARM *sparm) {
    /* Returns a feature vector describing the match between pattern x
       and label y. The feature vector is returned as a list of
       SVECTOR's. Each SVECTOR is in a sparse representation of pairs
       <featurenumber:featurevalue>, where the last pair has
       featurenumber 0 as a terminator. Featurenumbers start with 1 and
       end with sizePsi. Featuresnumbers that are not specified default
       to value 0. As mentioned before, psi() actually returns a list of
       SVECTOR's. Each SVECTOR has a field 'factor' and 'next'. 'next'
       specifies the next element in the list, terminated by a NULL
       pointer. The list can be though of as a linear combination of
       vectors, where each vector is weighted by its 'factor'. This
       linear combination of feature vectors is multiplied with the
       learned (kernelized) weight vector to score label y for pattern
       x. Without kernels, there will be one weight in sm.w for each
       feature. Note that psi has to match
       find_most_violated_constraint_???(x, y, sm) and vice versa. In
       particular, find_most_violated_constraint_???(x, y, sm) finds
       that ybar!=y that maximizes psi(x,ybar,sm)*sm.w (where * is the
       inner vector product) and the appropriate function of the
       loss + margin/slack rescaling method. See that paper for details. */
    SVECTOR *fvec = new SVECTOR[1];
    fvec->words = new WORD[model->sizePsi +1];
    int bx = L.y[0], by = L.y[1];
    int bw = model->bw;
    int bh = model->bh;

    WORD *words = fvec->words;
    for(int i = 0; i< model->sizePsi; i++)
        words[i].wnum = i +1;
    for(int i =0; i < bh; i++)
        for(int j = 0; j < bw; j++)
            words[(i)*bw+j].weight = x.pword ->val[by-1+i][bx-1+j];
    words[model->sizePsi].wnum = 0;
    words[model->sizePsi].weight = 0.0;

    fvec->factor =1.0;
    fvec->next = NULL;
    fvec->userdefined = NULL;
    return (fvec);
}

double      loss(LABEL y, LABEL ybar, STRUCT_LEARN_PARM *sparm)
{
  /* loss for correct label y and predicted label ybar. The loss for
     y==ybar has to be zero. sparm->loss_function is set with the -l option. */
    return (1-iou(y.y, ybar.y, sparm))*100;
}

double iou(int *yt, int *ypre, STRUCT_LEARN_PARM *sparm){
    int bw = sparm->bw; int bh = sparm->bh;
    int lx1 = yt[0]; int rx1 = lx1 + bw;
    int ly1 = yt[1]; int ry1 = ly1 + bh;
    int lx2 = ypre[0]; int rx2 = lx2 + bw;
    int ly2 = ypre[1]; int ry2 = ly2 + bh;
    double SI= MAX(0,MIN(rx1,rx2) - MAX(lx1,lx2))*MAX(0,MIN(ry1,ry2) - MAX(ly1,ly2));
    int S1 = bw*bh, S2 = bw*bh;
    return (SI/(S1+S2-SI));
}



int         finalize_iteration(double ceps, int cached_constraint,
			       SAMPLE sample, STRUCTMODEL *sm,
			       CONSTSET cset, double *alpha, 
			       STRUCT_LEARN_PARM *sparm)
{
  /* This function is called just before the end of each cutting plane iteration. ceps is the amount by which the most violated constraint found in the current iteration was violated. cached_constraint is true if the added constraint was constructed from the cache. If the return value is FALSE, then the algorithm is allowed to terminate. If it is TRUE, the algorithm will keep iterating even if the desired precision sparm->epsilon is already reached. */
  return(0);
}

void        print_struct_learning_stats(SAMPLE sample, STRUCTMODEL *sm,
					CONSTSET cset, double *alpha, 
					STRUCT_LEARN_PARM *sparm)
{
  /* This function is called after training and allows final touches to
     the model sm. But primarly it allows computing and printing any
     kind of statistic (e.g. training error) you might want. */
}

void        print_struct_testing_stats(SAMPLE sample, STRUCTMODEL *sm,
				       STRUCT_LEARN_PARM *sparm,
				       STRUCT_TEST_STATS *teststats)
{
  /* This function is called after making all test predictions in
     svm_struct_classify and allows computing and printing any kind of
     evaluation (e.g. precision/recall) you might want. You can use
     the function eval_prediction to accumulate the necessary
     statistics for each prediction. */

  printf("\nAverage iou: %lf", teststats->total_iou / sample.n);
  printf("\nprecision: %lf", teststats->truth / sample.n);
}

void        eval_prediction(long exnum, EXAMPLE ex, LABEL ypred, 
			    STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, 
			    STRUCT_TEST_STATS *teststats)
{
  /* This function allows you to accumlate statistic for how well the
     predicition matches the labeled example. It is called from
     svm_struct_classify. See also the function
     print_struct_testing_stats. */
    if(exnum == 0) {
        /* this is the first time the function is called. So initialize the teststats */
        teststats->total_iou = 0;
        teststats->truth = 0;
    }
    int *yt, *yp;
    yt = new int[2];
    yp = new int[2];
    yt[0] = ex.y.y[0]; yt[1] = ex.y.y[1];
    yp[0] = ypred.y[0]; yp[1] = ypred.y[1];

    double a = iou(yt, yp, sparm);
    printf("\npre: %d %d gt: %d %d iou: %lf ", yp[0], yp[1], yt[0], yt[1], a);
    fflush(stdout);
    teststats->total_iou += a;
    if(a > 0.5)
        teststats->truth += 1;
}

void        write_struct_model(char *file, STRUCTMODEL *sm, 
			       STRUCT_LEARN_PARM *sparm) {
    /* Writes structural model sm to file file. */
    FILE *fl;
    MODEL *model = sm->svm_model;

    if ((fl = fopen(file, "w")) == NULL) {
        perror(file);
        exit(1);
    }

    fprintf(fl, "%ld # sizePsi\n", sm->sizePsi);
    fprintf(fl, "%d # width of bounding box\n", sm->bw);
    fprintf(fl, "%d # height of bounding box\n", sm->bh);

    for (long i = 0; i < sm->sizePsi; i++) {
        fprintf(fl, "%.8lf ", model->lin_weights[i]);
    }

    fprintf(fl, "\n");
    fclose(fl);

}

STRUCTMODEL read_struct_model(char *file, STRUCT_LEARN_PARM *sparm)
{
  /* Reads structural model sm from file file. This function is used
     only in the prediction module, not in the learning module. */
    FILE *fl;
    STRUCTMODEL sm;
    long sizePsi;
    long max_words_doc;
    long max_docs;
    long ll;
    long pos;
    char *line;
    double num;
    long i;
    int bw;
    int bh;

    MODEL *model=(MODEL*)malloc(sizeof(MODEL)*1);
    nol_ll(file,&max_docs, &max_words_doc, &ll);
    ll+=10;

    if((fl=fopen(file, "r"))==NULL)
    {
        perror(file);
        exit(1);
    }
    fscanf(fl,"%ld%*[^\n]\n",&sizePsi);
    fscanf(fl,"%d%*[^\n]\n",&bw);
    fscanf(fl,"%d%*[^\n]\n",&bh);

    sm.sizePsi = sizePsi;

    WORD *words;
    words = (WORD *)malloc(sizeof(WORD)*sizePsi+1);
    line=(char*)malloc(sizeof(char)*ll);
    fgets(line,(int)ll,fl);
    pos=0;
    i=0;
    while(line[pos]!='\n')
    {
        sscanf(line+pos,"%lf",&num);
        words[i].weight = num;
        words[i].wnum = i+1;
        i++;
        while(line[pos]!=' ') pos++;
        while(line[pos]==' ') pos++;

    }
    words[i].wnum=0;
    fclose(fl);
    model->kernel_parm.kernel_type = 0;
    model->sv_num = 2;
    model->b = 0;
    model->totwords = sizePsi;
    model->supvec = (DOC **)my_malloc(sizeof(DOC *)*model->sv_num);
    model->alpha = (double *)my_malloc(sizeof(double)*model->sv_num);
    model->alpha[1] = 1;
    model->supvec[1] = create_example(-1,0,0,0.0,
                                      create_svector(words,NULL,1.0));
    model->supvec[1]->fvec->kernel_id=0;
    model->index=NULL;
    model->lin_weights=NULL;
    free(words);
    free(line);
    sm.svm_model = model;
    sm.bw = bw;
    sm.bh = bh;
    sm.w = NULL;
    sparm->bw = bw;
    sparm->bh = bh;
    return sm;
}

void        write_label(FILE *fp, LABEL y)
{
  /* Writes label y to file handle fp. */
} 

void        free_pattern(PATTERN x) {
  /* Frees the memory of x. */
}

void        free_label(LABEL y) {
  /* Frees the memory of y. */
}

void        free_struct_model(STRUCTMODEL sm) 
{
  /* Frees the memory of model. */
  /* if(sm.w) free(sm.w); */ /* this is free'd in free_model */
  if(sm.svm_model) free_model(sm.svm_model,1);
  /* add free calls for user defined data here */
}

void        free_struct_sample(SAMPLE s)
{
  /* Frees the memory of sample s. */
  int i;
  for(i=0;i<s.n;i++) { 
    free_pattern(s.examples[i].x);
    free_label(s.examples[i].y);
  }
  free(s.examples);
}

void        print_struct_help()
{
  /* Prints a help text that is appended to the common help text of
     svm_struct_learn. */
  printf("         --* string  -> custom parameters that can be adapted for struct\n");
  printf("                        learning. The * can be replaced by any character\n");
  printf("                        and there can be multiple options starting with --.\n");
}

void         parse_struct_parameters(STRUCT_LEARN_PARM *sparm)
{
  /* Parses the command line parameters that start with -- */
  int i;

  for(i=0;(i<sparm->custom_argc) && ((sparm->custom_argv[i])[0] == '-');i++) {
    switch ((sparm->custom_argv[i])[2]) 
      { 
      case 'a': i++; /* strcpy(learn_parm->alphafile,argv[i]); */ break;
      case 'e': i++; /* sparm->epsilon=atof(sparm->custom_argv[i]); */ break;
      case 'k': i++; /* sparm->newconstretrain=atol(sparm->custom_argv[i]); */ break;
      default: printf("\nUnrecognized option %s!\n\n",sparm->custom_argv[i]);
	       exit(0);
      }
  }
}

void        print_struct_help_classify()
{
  /* Prints a help text that is appended to the common help text of
     svm_struct_classify. */
  printf("         --* string -> custom parameters that can be adapted for struct\n");
  printf("                       learning. The * can be replaced by any character\n");
  printf("                       and there can be multiple options starting with --.\n");
}

void         parse_struct_parameters_classify(STRUCT_LEARN_PARM *sparm)
{
  /* Parses the command line parameters that start with -- for the
     classification module */
  int i;

  for(i=0;(i<sparm->custom_argc) && ((sparm->custom_argv[i])[0] == '-');i++) {
    switch ((sparm->custom_argv[i])[2]) 
      { 
      /* case 'x': i++; strcpy(xvalue,sparm->custom_argv[i]); break; */
      default: printf("\nUnrecognized option %s!\n\n",sparm->custom_argv[i]);
	       exit(0);
      }
  }
}

