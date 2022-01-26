#include <iostream>
#include <vector>

using namespace std;

void getmatch(vector <vector<double> >& DistMatrix, vector<int>& Assignment)
{
    Assignment.clear();
    vector<int> idx_vec;
    vector<double> dis_vec;



    // association
    for(int t = 0; t < DistMatrix.size(); t++)
    {
        double minDis = 100;
        int idx = -1;

        for(int d = 0; d < DistMatrix[0].size(); d++)
        {
            if(DistMatrix[t][d] > minDis)
                continue;

            idx = d;
            minDis = DistMatrix[t][d];
        }

        idx_vec.push_back(idx);
        dis_vec.push_back(minDis);
    }


    for(int source = 0; source < DistMatrix.size(); source++)
    {
        for(int target = source+1; target < DistMatrix.size(); target++)
        {
            if( idx_vec[source]!=idx_vec[target] || idx_vec[source]==-1 || idx_vec[target]==-1)
                continue;

            if( dis_vec[source] > dis_vec[target] )
                idx_vec[source] = -1;
            else if( dis_vec[source] < dis_vec[target] )
                idx_vec[target] = -1;
        }
    }


    vector<int> tmp_idx_vec = idx_vec;
    vector<double> tmp_dis_vec = dis_vec;

    // re:association
    for(int t = 0; t < DistMatrix.size(); t++)
    {
        if(idx_vec[t] != -1)
            continue;

        double minDis = 100;
        int idx = -1;

        for(int d = 0; d < DistMatrix[0].size(); d++)
        {
            vector<int>::iterator it = std::find(idx_vec.begin(), idx_vec.end(), d);
            if(it != idx_vec.end())
                continue;

            if(DistMatrix[t][d] > minDis)
                continue;

            idx = d;
            minDis = DistMatrix[t][d];
        }

        tmp_idx_vec[t] = idx;
        tmp_dis_vec[t] = minDis;
    }
    idx_vec = tmp_idx_vec;
    dis_vec = tmp_dis_vec;


    for(int source = 0; source < DistMatrix.size(); source++)
    {
        for(int target = source+1; target < DistMatrix.size(); target++)
        {
            if( idx_vec[source]!=idx_vec[target] || idx_vec[source]==-1 || idx_vec[target]==-1)
                continue;

            if( dis_vec[source] > dis_vec[target] )
                idx_vec[source] = -1;
            else if( dis_vec[source] < dis_vec[target] )
                idx_vec[target] = -1;
        }
    }


    //
    Assignment = idx_vec;
}
