#include<bits/stdc++.h>
#include<random>
using namespace std;

#define TRAINING_SIZE 32561
#define TESTING_SIZE 16281
#define ENSEMBLE_SIZE 1000
#define SAMPLING_FRACTION 0.632
mt19937 rng;
struct node
{
	string decisionattribute;
	map<string,node*> children;
};

int compare(vector<int> first,vector<int> second)
{
	return first[0]<second[0];
}
/**** given positive and negative values - calculates entropy of a system *****/
double entropy(int pos,int neg)
{
	if(pos == 0 || neg == 0)
		return 0;
	double pp,pn,res,tot;
	tot=pos+neg;
	pp=pos/tot;
	pn=neg/tot;
	res=(pp*log10(pp))+(pn*log10(pn));
	res/=log10(2);
	res*=-1.0;
	return res;
}

/*****calculates entropy of system ********/
double sys_entropy(vector<map<string,string> > dataset, string target)
{
	int i;
	int pos=0,neg=0;
	double res;
 	for(i=0;i<dataset.size();i++)
	{
		if(stoi(dataset[i][target])==1)
			pos++;
		else
			neg++;
	}
	res = entropy(pos,neg);
	return res;	
}

/*****calculates entropy of system with respect to the given discrete attribute ********/
double entropy_discrete(vector<map<string,string> > dt, string attribute,string target)
{
	int sum=0;
	double ent=0;
	map<string,int> pos;
	map<string,int> neg;
	set<string> attributevals;
 	for(int i=0;i<dt.size();i++)
	{
		attributevals.insert(dt[i][attribute]);
		if(stoi(dt[i][target])==1)
		   pos[dt[i][attribute]]++;
		else
		   neg[dt[i][attribute]]++;						 
	}
	for(set<string>::iterator it = attributevals.begin(); it!=attributevals.end();++it)
	{
		int posvals = pos[*it];
		int negvals = neg[*it];
		double tempentropy = entropy(posvals,negvals);
		ent = ent + (((double)posvals+(double)negvals)*tempentropy);
	}
	ent = ent/((double)dt.size());
	return ent;
}

/*****calculates entropy of system with respect to the given continuous attribute and also the split for continuous attribute entropy calculation ********/
vector<double> entropy_continuous(vector<map<string,string> > dt, string attribute, string target)
{

	vector<vector<int> > split;
	for(int i=0;i<dt.size();i++)
	{
		vector<int> temp(2);
		split.push_back(temp);
		split[i][0]=stoi(dt[i][attribute]);
	    split[i][1]=stoi(dt[i][target]);
	}

	stable_sort(split.begin(),split.end(),compare);

	vector<vector<double> > en_cont;
	int size = dt.size();
	int k,m;
	double min,min_idx,lp,ln,rp,rn;
  	k=0,lp=0,ln=0,rp=0,rn=0;
  	for(int i=1;i<size;i++)
  	{
  		if(split[i][1]!=split[i-1][1])
  		{  			
  			lp=0,ln=0,rp=0,rn=0;
  			vector<double> temp(2);
  			en_cont.push_back(temp);
  			en_cont[k][0]=(split[i][0]+split[i-1][0])/2;
  			for(m=0;m<i;m++)
  			{
  				if(split[m][1]==0)
  					ln++;
  				else
  					lp++;
  			}
  			for(m=i;m<size;m++)
  			{
  				if(split[m][1]==0)
  					rn++;
  				else
  					rp++;
  			}
  			m=lp+ln+rn+rp;
  			en_cont[k++][1]=(((lp+ln)*entropy(lp,ln))+((rp+rn)*entropy(rp,rn)))/m;
  		}
  	}
    min=size;
	for(int i=1;i<k;i++)
	{
		if(en_cont[i][1]<min)
		{
			min=en_cont[i][1];
			min_idx=i;
		}
	}
	return en_cont[min_idx];
}

void populaterandom(vector<map<string,string> > &trainset, vector<map<string,string> > &randtrainset) //populates random training examples for training the decision trees
{
	rng.seed(random_device()());
    uniform_int_distribution<int> uint_dist(0,trainset.size()-1);
    int iter = ceil((double)trainset.size()*SAMPLING_FRACTION);
    #pragma omp parallel for
		for(int i=0;i<iter;i++) //sampling from the training set
			randtrainset.push_back(trainset[uint_dist(rng)]);
}

node* randomforest(set<string> disc, set<string> cont,vector<map<string,string> > trainset,string target,vector<string> attributelist)
{
	if(trainset.size() == 0)
	{
		node *neg = new node;
		neg->decisionattribute = "0";
		return neg;
	}
	if(sys_entropy(trainset,target)==0)
	{
		node *leaf = new node;
		leaf->decisionattribute = trainset[0][target];
		return leaf;
	}
	set<string> discrete;
	set<string> continuous;
	rng.seed(random_device()());
    uniform_int_distribution<int> uint_dist(0,attributelist.size()-1);
    set<string> chosen;
	for(int j=0;j<ceil(sqrt(attributelist.size()));j++)
	{
		int rand = uint_dist(rng);
		string attributename = attributelist[rand];
		chosen.insert(attributename);
		if(disc.find(attributename)!=disc.end()) //discrete attribute
		{
			discrete.insert(attributename);
		}
		else if(cont.find(attributename)!=cont.end()) //continuous attribute
		{
			continuous.insert(attributename);
		}
		else
		{
			cout<<"ERROR\n";
		}
	}
	//finding attribute with minimum entropy
	string minentattribute;
	double minen = DBL_MAX;
	for(set<string>::iterator i=discrete.begin();discrete.size()!=0 && i!=discrete.end();++i)
	{
		if(entropy_discrete(trainset,*i,target) <= minen)
		{
			minentattribute = *i;
			minen = entropy_discrete(trainset,*i,target);
		}
	}
	for(set<string>::iterator i=continuous.begin();continuous.size()!=0 && i!=continuous.end();++i)
	{
		vector<double> temp = entropy_continuous(trainset,*i,target);
		if(temp[1] <= minen)
		{
			minentattribute = *i;
			minen = temp[1];
		}
	}		
	//create new decision node
	node *decisionnode = new node;
	decisionnode->decisionattribute = minentattribute;
	
	for(int i=0;i<attributelist.size();i++)
	{
		if(attributelist[i]==minentattribute)
		{
			attributelist.erase(attributelist.begin()+i);
			break;
		}
	}
	if(discrete.find(minentattribute)!=discrete.end())//the decision attribute is discrete valued
	{
		map<string,vector<map<string,string> > > m;
		disc.erase(disc.find(minentattribute));//removing chosen attribute from list of discrete attributes
		for(int i=0;i<trainset.size();i++)
		{
			//create the vector of details for each attribute value
			m[trainset[i][minentattribute]].push_back(trainset[i]);
		}
		for(map<string,vector<map<string,string> > >::iterator it = m.begin();it!=m.end();++it)
		{
			//create sub trees for every attribute value
			decisionnode->children[it->first] = randomforest(disc,cont,it->second,target,attributelist);
		}
	}
	else if(continuous.find(minentattribute)!=continuous.end()) //the decision attribute is continuous valued
	{
		map<string,vector<map<string,string> > > m;
		double spliton = (entropy_continuous(trainset,minentattribute,target))[0];
		cont.erase(cont.find(minentattribute));////removing chosen attribute from list of continuous attributes
		for(int i=0;i<trainset.size();i++)
		{
			if(stoi(trainset[i][minentattribute])<spliton)
				m["lesser than "+to_string(spliton)].push_back(trainset[i]);
			else
				m["greater than or equal to "+to_string(spliton)].push_back(trainset[i]);
		}
		for(map<string,vector<map<string,string> > >::iterator it = m.begin();it!=m.end();++it)
		{
			//create sub trees for every attribute value
			decisionnode->children[it->first] = randomforest(disc,cont,it->second,target,attributelist);
		}
	}
	else
	{
		decisionnode->decisionattribute = "0";
	}
	return decisionnode;
}

void printtree(node *root,int level,map<int, vector<string> > &m)
{
	m[level].push_back(root->decisionattribute);
	for(map<string,node*>::iterator it = root->children.begin();it!=root->children.end();++it)
	{
		m[level].push_back(it->first);
	}
	for(map<string,node*>::iterator it = root->children.begin();it!=root->children.end();++it)
	{
		printtree(it->second,level+1,m);
	}
}

int classify(map<string,string> test, string target, node *root,set<string> discrete,set<string> continuous)
{
	if(root->children.size()==0)
		return stoi(root->decisionattribute);
	string decisionattribute = root->decisionattribute;

	if(discrete.find(decisionattribute)!=discrete.end()) //decision attribute is discrete valued
	{
		if(root->children.find(test[decisionattribute])!=root->children.end())
			return classify(test,target,root->children[test[decisionattribute]],discrete,continuous);
		else
			return 0;
	}
	else if(continuous.find(decisionattribute)!=continuous.end())
	{
		double spliton;
		//getting the value of the split value
		if(root->children.size()==2)
			spliton = stod((root->children.begin())->first.substr(25));
		else
		{
			string temp = (root->children.begin())->first;
			if(temp[0] == 'l')
				spliton = stod((root->children.begin())->first.substr(12));
			else
				spliton = stod((root->children.begin())->first.substr(25));
		}
		if(stoi(test[decisionattribute]) < spliton)
		{
			if(root->children.find("lesser than " + to_string(spliton))!=root->children.end())
				return classify(test,target,root->children["lesser than " + to_string(spliton)],discrete,continuous);
			else
				return 0;
		}
		else
		{
			if(root->children.find("greater than or equal to " + to_string(spliton))!=root->children.end())
				return classify(test,target,root->children["greater than or equal to " + to_string(spliton)],discrete,continuous);
			else
				return 0;
		}
	}
}

int main()
{
	vector<map<string,string> > trainset;
	set<string> discrete;
	set<string> continuous;
	vector<string> attributelist;
	discrete.insert("workclass");
	discrete.insert("education");
	discrete.insert("marital_status");
	discrete.insert("occupation");
	discrete.insert("relationship");
	discrete.insert("race");
	discrete.insert("sex");
	discrete.insert("native_country");
	continuous.insert("age");
	continuous.insert("fnlwgt");
	continuous.insert("education_num");
	continuous.insert("capital_gain");
	continuous.insert("capital_loss");
	continuous.insert("hours_per_week");

	attributelist.push_back("workclass");
	attributelist.push_back("education");
	attributelist.push_back("marital_status");
	attributelist.push_back("occupation");
	attributelist.push_back("relationship");
	attributelist.push_back("race");
	attributelist.push_back("sex");
	attributelist.push_back("native_country");
	attributelist.push_back("age");
	attributelist.push_back("fnlwgt");
	attributelist.push_back("education_num");
	attributelist.push_back("capital_gain");
	attributelist.push_back("capital_loss");
	attributelist.push_back("hours_per_week");
	string target = "income";
	fstream f("train.txt",ios::in);	
	for(int i=0;i<TRAINING_SIZE;i++)
	{
		map<string,string> temp;
		f>>temp["age"]>>temp["workclass"]>>temp["fnlwgt"]>>temp["education"]>>temp["education_num"]>>temp["marital_status"]>>temp["occupation"]>>temp["relationship"]>>temp["race"]>>temp["sex"]>>temp["capital_gain"]>>temp["capital_loss"]>>temp["hours_per_week"]>>temp["native_country"]>>temp["income"];
		int check=0;
		for(map<string,string>::iterator it=temp.begin();it!=temp.end();++it)
		{
			if(it->second=="?")
			{
				check = 1;
				break;
			}
		}
		if(!check)
			trainset.push_back(temp);	
	}
	vector<node*> ensemble;
	#pragma omp parallel for
	    for(int i=0;i<ENSEMBLE_SIZE;i++)
	    {
	    	vector<map<string,string> > randtrainset;
	    	populaterandom(trainset,randtrainset);    	
	    	ensemble.push_back(randomforest(discrete,continuous,randtrainset,target,attributelist));
	    	cout<<"Decision tree "<<i+1<<" trained"<<endl;
	    }
	cout<<"Training complete\n";
	fstream g("test.txt",ios::in);
	int correct = 0, total = 0;
	for(int i=0;i<TESTING_SIZE;i++)
	{
		map<string,string> temp;
		g>>temp["age"]>>temp["workclass"]>>temp["fnlwgt"]>>temp["education"]>>temp["education_num"]>>temp["marital_status"]>>temp["occupation"]>>temp["relationship"]>>temp["race"]>>temp["sex"]>>temp["capital_gain"]>>temp["capital_loss"]>>temp["hours_per_week"]>>temp["native_country"]>>temp["income"];
		int posvotes,negvotes;
		posvotes = negvotes = 0;
		vector<double> classifications(ensemble.size());
		#pragma omp parallel for
			for(int j=0;j<ensemble.size();j++)
				classifications[j] = classify(temp,target,ensemble[j],discrete,continuous);
		for(int j=0;j<ensemble.size();j++)
		{
			if(classifications[j])
				posvotes++;
			else
				negvotes++;
		}
		int res = posvotes>=negvotes ? 1:0;
		if(res == stoi(temp[target]))
			correct++;
		total++;
	}
	cout<<"Accuracy over test set = "<<(double)correct/(double)total<<endl;
	return 0;
}