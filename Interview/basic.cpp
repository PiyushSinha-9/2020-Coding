/*
* @Author: Piyush Sinha
* @Date:   2020-03-07 21:33:06
* @Last Modified by:   Piyush Sinha
* @Last Modified time: 2020-04-20 10:24:02
*/


////////////////////////////  concept of finding point where array is rotated...
///
///  min in rotated array   
///  
///  1st approach for generally unique elements..
///  class Solution {
public:
    
    int helper(vector<int>& nums,int start, int end){
        if(start>=end){
            return -1;
        }

        int mid=(start+end)/2;
        if(nums[mid]>nums[mid+1]){
            return mid;
        }


        if(nums[mid]<nums[start]){
            return helper(nums,start,mid);
        }else{
            return helper(nums,mid+1,end);
        }
        
    }
    
    
    
    int findMin(vector<int>& nums) {
        int n=nums.size();
        int pivot=helper(nums,0,n-1);
        if(pivot==-1){
            return nums[0];
        }
        else{
            return nums[pivot+1];
        }
        
        
    }
};  

////    2nd approach best one.  works with duplicate elements too.
class Solution {
public:
    int helper(vector<int>& nums,int start,int end){
        if(start>=end){
            return -1;
        }
        int mid=(start+end)/2;
        if(nums[mid]>nums[mid+1]){
            return mid;
        }
        
        if(nums[mid]<nums[start]){
            return helper(nums,start,mid);
        }else if(nums[mid]>nums[start]){
            return helper(nums,mid+1,end);
        }else{
            int sol1=helper(nums,start,mid);
            int sol2=helper(nums,mid+1,end);
            
            if(sol1!=-1 and nums[sol1]>nums[sol1+1] ){
                return sol1;
            }else{
                return sol2;
            }
            
        }
        
    }
    int findMin(vector<int>& nums) {
        int n=nums.size();
        int pivot=helper(nums,0,n-1);
        if(pivot==-1){
            return nums[0];
        }else{
            return nums[pivot+1];
        }
    }
};




////////////////////////////////////////   peak index of element in rotated sorted array
///
	int n;
    int helper(vector<int> & nums, int start,int end){
        if(start>=end){
            return -1;
        }
        
        int mid=(start+end)/2;
        if(mid>0 and nums[mid]>nums[mid+1] and nums[mid]>nums[mid-1]){
            return mid;
        }
        
        if(end==n-1 and nums[end]>nums[end-1]){
            return end;
        }
        
        if(start==0 and nums[start]>nums[start+1]){
            return start;
        }
        
        int sol1=helper(nums,start,mid);
        int sol2=helper(nums,mid+1,end);
        
        if(sol1!=-1){
            return sol1;
        }
        else{
            return sol2;
        }
        
        
        
    }
    
    int findPeakElement(vector<int>& nums) {
        n=nums.size();
        if(n==1){
            return 0; 
        }
        return helper(nums,0,n-1);
    }
//////////////////////////////////////////////////////////////////////////////////// 

///////////////////////////////////////   house robbing linearly placed houses
///
int rob(vector<int>& nums) {
        int n=nums.size(),ans=0;

        if(n==0){
            return 0;
        }
        if(n==1){
            return nums[0];
        }
        
        int *dp=new int[n];
        dp[0]=nums[0];
        dp[1]=max(nums[0],nums[1]);
        
        for(int i=2;i<n;i++){
            dp[i]=max(nums[i]+dp[i-2],dp[i-1]);       
        }
        
        ans=dp[n-1];
        delete [] dp;
        return ans;
        
    }

 /////////////// 							circular loop arranged.
 ///
	class Solution {
public:
    int houseRobber(vector<int>& nums,int n) {
        if(n==0){
            return 0;
        }
        if(n==1){
            return nums[0];
        }
        
        int *dp=new int[n];
        dp[0]=nums[0];
        dp[1]=max(nums[1],nums[0]);
        
        for(int i=2;i<n;i++){
            dp[i]=max(nums[i]+dp[i-2],dp[i-1]);
        }
        
        return dp[n-1];
    }
    
    int rob(vector<int>& nums) {
        int n=nums.size();
        if(n==0){
            return 0;
        }
        if(n==1){
            return nums[0];
        }

        int store=nums[n-1];
        nums.pop_back();
        int second_part=houseRobber(nums,n-1);
        
        nums.push_back(store);
        nums.erase(nums.begin());
        int first_part=houseRobber(nums,n-1);
        
        nums.insert(nums.begin(),store);
        
        return max(second_part,first_part);
        
    }
};

//////////////////////							housing 3 
///
	 pair<int,int> helper(TreeNode *root){
        if(root==NULL){
            return make_pair(0,0);
        }
        
        pair<int,int> ls=helper(root->left);
        pair<int,int> rs=helper(root->right);
        
        pair<int,int> nodeMax;
        nodeMax.first=root->val+ls.second+rs.second;
        nodeMax.second=max(ls.first,ls.second)+max(rs.first,rs.second);
        
        return nodeMax;
        
    }
    
    
    int rob(TreeNode* root) {
        pair<int,int> ans= helper(root);
        return max(ans.first,ans.second);
    }

    // help https://www.youtube.com/watch?v=bEZxjZCY618

 //////////////////////////////////////////////////////////////////////////////////// 
 ///
 ///
 
///////////////////////////// Find First and Last Position of Element in Sorted Array
 		int n;
    int lower_bound(vector<int>& nums,int target,int start,int end){
        if(start>end){
            return -1; 
        }
        if(start==end){
            if(nums[start]==target)
            return start;
            else return -1;
        }
        int mid=(start+end)/2;
        if(mid>0 and nums[mid]==target and nums[mid]>nums[mid-1]){
            return mid;
        }
        
        if(nums[mid]>=target){
            return lower_bound(nums,target,start,mid);
        }else{
            return lower_bound(nums,target,mid+1,end);
        }
        
    }
    
    int upper_bound(vector<int>& nums,int target,int start,int end){
        if(start>end){
            return -1; 
        }
        
        if(start==end){
            if(nums[start]==target){
                return start;
            }else{
                return -1;
            }
        }
        
        int mid=(start+end)/2;
        if(mid<n-1 and nums[mid]==target and nums[mid]<nums[mid+1]){
            return mid;
        }
        
        
        if(nums[mid]>target){
            return upper_bound(nums,target,start,mid);
        }else{
            return upper_bound(nums,target,mid+1,end);
        }
        
    }
    
    
    vector<int> searchRange(vector<int>& nums, int target) {
        vector<int> ans;
        n=nums.size();
        ans.push_back(lower_bound(nums,target,0,n-1));
        ans.push_back(upper_bound(nums,target,0,n-1));
        return ans;
    }


  //////////////////////////////////////////////////////////////////////////////////// 
  ///
  //////////////                     35. Search Insert Position
  ///
  	 int searchInsert(vector<int>& nums, int target) {
        int n=nums.size();
        return helper(nums,target,0,n-1);
    }
    
    int helper(vector<int>& nums,int target,int start,int end){
        if(start>end){
            return -1;
        }
        
        if(start==end){
            if(nums[start]<target){
                return start+1;
            }else{
                return start;
            }
        }
        
        int mid=(start+end)/2;
        
        if(nums[mid]<target and nums[mid+1]>=target){
            return mid+1;
        }
        
        if(nums[mid]>=target){
            return helper(nums,target,start,mid);
        }else{
            return helper(nums,target,mid+1,end);
        }
    }


    /////////////////////////						normal binary search or -1 if not exist
    ///
    int search(vector<int>& nums, int target) {
        int n=nums.size();
        return helper(nums,target,0,n-1);
    }
    
    int helper(vector<int>& nums, int target,int start,int end){
        if(start>end){
            return -1;
        }
        
        if(start==end){
            if(nums[start]==target){
                return start;
            }
            else{
                return -1;
            }
        }
        
        int mid=(start+end)/2;
        if(nums[mid]==target){
            return mid;
        }
        
        if(nums[mid]>target){
            return helper(nums,target,start,mid);
        }else{
             return helper(nums,target,mid+1,end);
        }
    }

 //////////////////////////////////////////////////////////////////////////////////// 
 ///
 ///
/////////////////////					power of n or -n
double helper(double x,int n){
        if(n==0){
            return 1;
        }
        double val=helper(x,n/2);
        if(n%2!=0)
        {
            return val*val*x;
        }else{
            return val*val;
        }
    }
    double myPow(double x, int n) {
        if(n<0){
            return helper(1/x,(unsigned)(-1)*n);
        }else{
            return helper(x,n);
        }
        
    }


 //////////////////////////////////////////////////////////////////////////////////// 
 ///
 ///								//// subset with dulpicate
   set<vector<int>> ans1;
    void helper(vector<int>& nums,int n,vector<int> temp,int pos=0)
    {
        if(pos>=n){
            sort(temp.begin(),temp.end());
            ans1.insert(temp);
            return;
        }
        
        helper(nums,n,temp,pos+1);
        temp.push_back(nums[pos]);
        helper(nums,n,temp,pos+1);
    }
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        vector<vector<int>> ans;
        int n=nums.size();
        if(n==0){
            return ans;
        }
        vector<int> temp;
        helper(nums,n,temp);
        auto ptr=ans1.begin();
        while(ptr!=ans1.end()){
            ans.push_back(*ptr);
            ptr++;
        }
        return ans;
    }

  //////////////////////////////////////////////////////////////////////////////////// 
  ///
  ///
  //////////////////////					merging intervals
  ///
  static bool compare(vector<int> a,vector<int> b){
        if(a[0]<b[0]){
            return true;
        }else if(a[0]==b[0] and a[1]<b[1]){
            return true;
        }
        return false;
    }
    vector<vector<int>> merge(vector<vector<int>>& inter) {
        int n=inter.size();
        vector<vector<int>> ans;
        if(n==0){
            return ans;
        }
        sort(inter.begin(),inter.end(),compare);
        
        vector<int> top=inter[0];
        for(int i=1;i<n;i++){
            if(top[1]>=inter[i][0]){
                top[1]=max(top[1],inter[i][1]);
                continue;
            }
            else{
                ans.push_back(top);
                top=inter[i];
            }
        }
        ans.push_back(top);
        return ans;
    }

    ///////////////////////////////					insert intervals
    ///
    int n;
    static bool compare(vector<int> a,vector<int> b){
        if(a[0]<b[0]){
            return true;
        }else if(a[0]==b[0] and a[1]<b[1]){
            return true;
        }
        return false;
    }
    vector<vector<int>> merge(vector<vector<int>>& inter) {
        int n=inter.size();
        vector<vector<int>> ans;
        if(n==0){
            return ans;
        }
        //sort(inter.begin(),inter.end(),compare);
        
        vector<int> top=inter[0];
        for(int i=1;i<n;i++){
            if(top[1]>=inter[i][0]){
                top[1]=max(top[1],inter[i][1]);
                continue;
            }
            else{
                ans.push_back(top);
                top=inter[i];
            }
        }
        ans.push_back(top);
        return ans;
    }
    
    vector<vector<int>> insert(vector<vector<int>>& inter, vector<int>& newInterval) {
        n=inter.size();
        sort(inter.begin(),inter.end(),compare);
        auto ptr=inter.begin();
        while(ptr!=inter.end()){
            vector<int> yo=*ptr;
            if(yo[0]>newInterval[0]){
                break;
            }
            ptr++;
        }
    
        inter.insert(ptr,newInterval);
        
        n+=1;
        
        return merge(inter);
        
    }
  //////////////////////////////////////////////////////////////////////////////////// 
  ///
  ///
  ////////////////////					consecutive array element like 7-10 or 1-5
  ///
  ///   t o(n) and s o(1)
	  bool check(int *arr,int n){
	    int _max=INT_MIN,_min=INT_MAX;
	    for(int i=0;i<n;i++){
	        _max=max(arr[i],_max);
	        _min=min(arr[i],_min);
	    }
	    
	    if(_max-_min+1==n){
	        int j;
	        for(int i=0;i<n;i++){
	            
	            if(arr[i]>0){
	                j=arr[i]-_min;
	            }else{
	                j=-arr[i]-_min;
	            }
	            
	            if(arr[j]<0){
	                return false;
	            }
	            
	            arr[j]=-arr[j];
	        }
	        return true;
	    }else{
	        return false;
	    }
	    
}	

  //////////////////////////////////////////////////////////////////////////////////// 
  ///
  ///	
  //////////////////						subset sum is 0
  ///   t o(n)  s o(1)
  ///   
  			
		bool check(int *arr,int n){
		    unordered_set<int> dp;
		    int subsum=0;
		    for(int i=0;i<n;i++){
		        subsum+=arr[i];
		        if(subsum==0 or dp.find(subsum)!=dp.end()){
		            return true;
		        }
		        dp.insert(subsum);
		    }
		    return false;
		}

		int main() {
			ios_base::sync_with_stdio(false);
			cin.tie(0); cout.tie(0);
			
			int test;
			cin>>test;
			while(test--){
			    int n;
			    cin>>n;
			    int *arr=new int[n];
			    for(int i=0;i<n;i++){
			        cin>>arr[i];
			    }
			    
			    if(check(arr,n)){
			        cout<<"Yes";
			    }else{
			        cout<<"No";
			    }
			    
			    cout<<endl;
			}
			return 0;
		}			



///////////////////////////         largest subarray with array 0  (very important)


int maxLen(int A[], int n) {
    // Your code here       solving using general way
    unordered_map<int,int> dp;
    int k=0;
    int maxLen=0;
    int subset=0;
    
    for(int i=0;i<n;i++){
        subset+=A[i];
        if(subset==k){
            maxLen=max(maxLen,i+1);
        }
        
        if(dp.find(subset)==dp.end()){
            dp[subset]=i;
        }
        
        if(dp.find(subset-k)!=dp.end()){
            maxLen=max(maxLen,i-dp[subset-k]);
        }
        
    }
    
    return maxLen;
}



/////////////////////////////////////////////////////////			
///								Longest Sub-Array with Sum K

	int helper(int *arr, int n,int k){
	    
	    unordered_map<int,int> dp;
	    
	    int maxLen=0;
	    int subsum=0;
	    for(int i=0;i<n;i++){
	        subsum+=arr[i];
	        
	        if(subsum==k){
	            maxLen=max(maxLen,i+1);
	        }
	        
	        if(dp.find(subsum)==dp.end()){
	             dp[subsum]=i;
                 continue;
	        }
	        
	        if(dp.find(subsum-k)!=dp.end()){
	            maxLen=max(maxLen,i-dp[subsum-k]);
	        }
	    }
	    
	    return maxLen;
	}



  //////////////////////////////////////////////////////////////////////////////////// 
///
///						very very important   must study
///	https://www.geeksforgeeks.org/length-of-the-longest-substring-without-repeating-characters/
///
//////////////////////////// longest string substring without repeating char
///
	int lengthOfLongestSubstring(string s) {
	        int n=s.size();
	        if(n==0) return 0;

	        int nums_of_char=256;
	        int maxLen=1,tempLen=1,prev;
	        
	        int *dp=new int[nums_of_char];
	        for(int i=0;i<nums_of_char;i++){
	            dp[i]=-1;
	        }
	        
	        dp[s[0]]=0;

	        for(int i=1;i<n;i++){
	            prev=dp[s[i]];
	            
	            if(prev==-1 or i-tempLen>prev){
	                tempLen+=1;
	            }else{
	                maxLen=max(maxLen,tempLen);
	                tempLen=i-prev;
	            }
	            
	            dp[s[i]]=i;
	        }
	        maxLen=max(maxLen,tempLen);
	        
	        delete [] dp;
	        return maxLen;
	    }


  //////////////////////////////////////////////////////////////////////////////////// 

////////////////////////		reverse a stack using recursion
///

// using std::stack for 
// stack implementation 
stack<char> st; 

string ns; 

char insert_at_bottom(char x) 
{ 

	if(st.size() == 0) 
	   st.push(x); 
	else
	{ 	
		char a = st.top(); 
		st.pop(); 
		insert_at_bottom(x); 
		st.push(a); 
	} 
} 


char reverse() 
{ 
	if(st.size()>0) 
	{ 
		char x = st.top(); 
		st.pop(); 
		reverse(); 
		insert_at_bottom(x); 
	} 
} 

int main() 
{ 
	st.push('1'); 
	st.push('2'); 
	st.push('3'); 
	st.push('4'); 
	
	cout<<"Original Stack"<<endl; 

	cout<<"1"<<" "<<"2"<<" "
		<<"3"<<" "<<"4"
		<<endl; 
	reverse(); 
	cout<<"Reversed Stack"
		<<endl; 
	
	while(!st.empty()) 
	{ 
		char p=st.top(); 
		st.pop(); 
		ns+=p; 
	} 
	cout<<ns[3]<<" "<<ns[2]<<" "
		<<ns[1]<<" "<<ns[0]<<endl; 
	return 0; 
} 


  //////////////////////////////////////////////////////////////////////////////////// 
  ///
  ///
  ///										solving a soduko
  ///										
    
    
    bool check(vector<vector<int>> &grid,int val,int x,int y){
        
        // vertical
        for(int i=0;i<9;i++){
            if(grid[i][y]==val){
                return false;
            }
        }
        
        
        // horizontal
        for(int i=0;i<9;i++){
            if(grid[x][i]==val){
                return false;
            }
        }
        
        
        // in section
        
        int move_r=x%3,move_c=y%3;
        x-=move_r;
        y-=move_c;
        
        for(int i=x;i<x+3;i++){
            for(int j=y;j<y+3;j++){
                if(grid[i][j]==val){
                    return false;
                }
            }
        }
        
        
        return true;
        
    }
    
    bool helper(vector<vector<char>>& board,vector<vector<int>> &grid,int row,int col){
        
        if(row==9){
            return true;
        }
        
        if(col==9){
            if(helper(board,grid,row+1,0)){
                return true;
            }else{
                return false;
            }
        }
        
        if(board[row][col]!='.'){
            if(helper(board,grid,row,col+1)){
                return true;
            }else{
                return false;
            }
        }
        
        for(int i=1;i<=9;i++){
            if(check(grid,i,row,col)){
                grid[row][col]=i;
                if(helper(board,grid,row,col+1)){
                    return true;
                }
            }
        }
        
        grid[row][col]=0;
        return false;
    }
    
    void solveSudoku(vector<vector<char>>& board) {
        // vector<vector<char>> temp=board;
        vector<vector<int>> grid(9,vector<int>(9,0));
        
        for(int i=0;i<9;i++){
            for(int j=0;j<9;j++){
                if(board[i][j]=='.'){
                    continue;
                }
                grid[i][j]=board[i][j]-'0';
            }
        }
        
        helper(board,grid,0,0);
            
        
        for(int i=0;i<9;i++){
            for(int j=0;j<9;j++){
                board[i][j]=char(grid[i][j]+'0');
            }
        }
        
        return ;
        
    }


  //////////////////////////////////////////////////////////////////////////////////// 

//////////////////		EGG DROPPING PROBLEM   RECURSIVE SOLUTION

		map<pair<int,int>, int> dp;
		    int superEggDrop(int k, int n) {		//K EGGS N FLOORS
		        
		        if(n==0 or n==1 or k==1){
		            return n;
		        }
		        
		        if(dp[make_pair(k,n)]!=0){
		            return dp[make_pair(k,n)];
		        }
		        
		        int _min=INT_MAX,temp;
		        
		        for(int i=1;i<=n;i++){
		            temp=max(superEggDrop(k-1,i-1),superEggDrop(k,n-i));
		            _min=min(_min,temp);
		        }
		        
		        dp[make_pair(k,n)]=_min+1;
		        return _min+1;
		    }
/////////////////		DP Solution

///			t: o(k*n)  and s : o(n)
    def superEggDrop(self, K, N):
        dp = range(N+1)

        for k in xrange(2, K+1):
            dp2 = [0]
            x = 1
            for n in xrange(1, N+1):
                while x < n and max(dp[x-1], dp2[n-x]) > max(dp[x], dp2[n-x-1]):
                    x += 1
                dp2.append(1 + max(dp[x-1], dp2[n-x]))

            dp = dp2

        return dp[-1]

///////////////////			mathematics solution best solution

////		t: o(k*log(n))  and s: o(1)

    def superEggDrop(self, K, N):
        def f(x):
            ans = 0
            r = 1
            for i in range(1, K+1):
                r *= x-i+1
                r //= i
                ans += r
                if ans >= N: break
            return ans

        lo, hi = 1, N
        while lo < hi:
            mi = (lo + hi) // 2
            if f(mi) < N:
                lo = mi + 1
            else:
                hi = mi
        return lo


  //////////////////////////////////////////////////////////////////////////////////// 

        					///// tree 
class Result:
    valid: bool = True
    min: int = float('inf')
    max: int = float('-inf')
    sum: int = 0

class Solution(object):
    def maxSumBST(self, root: TreeNode) -> int:
        def dfs(node: TreeNode) -> Result:
            if not node:
                return Result()
            
            left, right = dfs(node.left), dfs(node.right)
            ans = Result(
                valid = left.valid and right.valid and left.max < node.val < right.min,
                min = min(left.min, node.val),
                max = max(right.max, node.val),
                sum = left.sum + right.sum + node.val
            )
            
            if ans.valid and ans.sum > self.best:
                self.best = ans.sum
            return ans
        
        self.best = 0
        dfs(root)
        return self.best




  ////////////////////////////////////////////////////////////////////////////////////
  ///
  ///			/////        move zeros at end inplace 
  ///			
	  void moveZeroes(vector<int>& nums) {
	        for (int lastNonZeroFoundAt = 0, cur = 0; cur < nums.size(); cur++) {
	        if (nums[cur] != 0) {
	            swap(nums[lastNonZeroFoundAt++], nums[cur]);
	        }
	    }
	    }

	    void moveZeroes(vector<int>& nums) {
        int cur=0,las_zer=0;
        int n=nums.size();
        for(;cur<n;cur++){
            if(nums[cur]!=0){
                swap(nums[las_zer],nums[cur]);
                las_zer+=1;
            }
        }
    }	


    //////////////////   useless question
    ///
    Given a string and an integer k, you need to reverse the first k characters for every
     2k characters counting from the start of the string. If there are less than k
      characters left, reverse all of them. If there are less than 2k but greater than 
      or equal to k characters, then reverse the first k characters and left the other 
      as original.

      class Solution:
    def reverseStr(self, s: str, k: int) -> str:
        s, l = list(s), len(s)
        for i in range(0, l, 2*k):
            end = i + min(l-i, k)
            for j in range(i, (i+end)//2):
                s[end-j+i-1], s[j] = s[j], s[end-j+i-1]
        return ''.join(s)

  //////////////////////////////////////////////////////////////////////////////////// 
  ///
  ///    							 Best Time to Buy and Sell Stock
  ///    
  		int maxProfit(vector<int>& pri) {
        int ans=0;
        int mini=INT_MAX,end=pri.size();
        for(int i=0;i<end;i++){
            if(pri[i]<mini){
                mini=pri[i];
            }else{
                ans=max(ans,pri[i]-mini);
            }
        }
        
        return ans;
    }


//////////////////////////////////////////////////////////////////////////////////// 
//////					kadane algorithm			continious subarray sum
///
		int test;
			cin>>test;
			while(test--){
			    int n;
			    cin>>n;
			    
			    int *arr=new int [n];
			    int maxi=INT_MIN;
			    int *dp=new int[n];
			    for(int i=0;i<n;i++){
			        cin>>arr[i];
			        if(i==0){
			            dp[0]=arr[i];
			            maxi=max(dp[i],maxi);
			        }else{
			            dp[i]=max(dp[i-1]+arr[i],arr[i]);
			            maxi=max(dp[i],maxi);
			        }
			    }
			   
			    delete [] dp;
			    delete [] arr;
			    
			    cout<<maxi<<endl;
			}
//////////////////////////				continuous subarray product
///
		
        int maxProduct(vector<int>& nums) {
        int n=nums.size();
        int cur_max,cur_min,prev_max,prev_min,ans=0;
        
        for(int i=0;i<n;i++){
            if(i==0){
                cur_max=nums[0];
                cur_min=nums[0];
                prev_max=nums[0];
                prev_min=nums[0];
                ans=nums[0];
            }
            else{
                cur_max=max({prev_max*nums[i],prev_min*nums[i],nums[i]});
                cur_min=min({prev_max*nums[i],prev_min*nums[i],nums[i]});
                ans=max(ans,cur_max);
                prev_max=cur_max;
                prev_min=cur_min;
            }
        }
        
        
        return ans;
        }	



    ////////////////////////////////////////////////////////////////////////////
    ///
    ///////////////				Product except self very important do use division
    	vector<int> productExceptSelf(vector<int>& nums) {
        // t o(n) and s o(n)
        
        int n=nums.size();
        vector<int> ans;
        if(n==1){
            ans.push_back(0);
            return ans;
        }
        
        int *dp1=new int[n];
        int *dp2=new int[n];
        
        dp1[0]=1;
        for(int i=1;i<n;i++){
            dp1[i]=dp1[i-1]*nums[i-1];
        }
        
        dp2[n-1]=1;
        for(int i=n-2;i>=0;i--){
            dp2[i]=dp2[i+1]*nums[i+1];
        }
        
        
        
        for(int i=0;i<n;i++){
            ans.push_back(dp1[i]*dp2[i]);
        }
        
        delete [] dp1;
        delete [] dp2;
        
        return ans;
    }

    ///////////////					2nd approach
    ///
    ///			 vector<int> productExceptSelf(vector<int>& nums) {
        // t-: o(n)  and space -: o(1)
        
        int n=nums.size(),hold;
        vector<int> ans(n,1);
        if(n==1){
            ans[0]=0;
            return ans;
        }
        hold=1;
        for(int i=1;i<n;i++){
            hold=hold*nums[i-1];
            ans[i]=hold;
        }
        
        hold=1;
        for(int i=n-2;i>=0;i--){
            hold*=nums[i+1];
            ans[i]*=hold;
        }
        
        
        return ans;
    }
  //////////////////////////////////////////////////////////////////////////////////// 
  ///
  ///
  ////////////////					tapping rain water problem
  ///

  		int trap(vector<int>& h) {
        int n=h.size();
        int *dpl=new int[n];
        int *dpr=new int[n];
        int ans=0;
        for(int i=0;i<n;i++){
            if(i==0){
                dpl[i]=h[i];
            }else{
                dpl[i]=max(dpl[i-1],h[i]);
            }
        }
        
        
        for(int i=n-1;i>=0;i--){
            if(i==n-1){
                dpr[i]=h[i];
            }
            else
            {
                dpr[i]=max(dpr[i+1],h[i]);
            }
        }
        
        for(int i=0;i<n;i++){
            ans+=min(dpl[i],dpr[i])-h[i];
        }
    
        delete [] dpl;
        delete [] dpr;
        
        return ans;
    }



    //////////////////////////////////////////////////////////////////////////////////// 
    ///
    ///
    /////////////////////////				largest area in histogram
    ///
    	

         stack<ll> s;
         ll area=INT_MIN,temp_area;
         s.push(0);

         for(ll i=1;i<n;i++){
             if(a[s.top()]<a[i]){
                 s.push(i);
             }
             else{
                 while(a[s.top()]>a[i]) {
                     ll top_ind = s.top();
                     s.pop();
                     if (s.empty()) {
                         temp_area = a[top_ind] * i;
                         if (temp_area > area) {
                             area = temp_area;
                         }
                     }
                     else {
                         temp_area=a[top_ind]*(i-s.top()-1);
                         if (temp_area > area) {
                             area = temp_area;
                         }
                     }

                     if(s.empty()){
                         break;
                     }

                 }

                 s.push(i);
             }
         }

        while(!s.empty()) {
            ll top_ind = s.top();
            s.pop();
            if (s.empty()) {
                temp_area = a[top_ind] * (n);
                if (temp_area > area) {
                    area = temp_area;
                }
            }
            else {
                temp_area=a[top_ind]*((n)-s.top()-1);
                if (temp_area > area) {
                    area = temp_area;
                }
            }
        }

///////////////////////////////////////////////////////////////////////////////////
///
///Given a string S and a string T, find the minimum window in S which will
/// contain all the characters in T in complexity O(n).			
///
         int max_size=256;
    
    bool check(int *&freqs,int *&freqt){
        for(int i=0;i<max_size;i++){
            if(freqs[i]<freqt[i]){
                return false;
            }
        }
        return true;
    }
    
    
    string minWindow(string s, string t) {
        int left=0,right=0;
        int n=s.size();
        int nt=t.size();
        string ans="";
        
        if(nt>n){
            return ans;
        }
        
        int min_left=0,min_right=-1,maxi=n;
        
        int *freqs=new int[max_size];
        int *freqt=new int[max_size];
        
        for(int i=0;i<max_size;i++){
            freqt[i]=0;
            freqs[i]=0;
        }
        
        for(int i=0;i<nt;i++){
            freqt[t[i]]+=1;    
        }
        
        for(int i=0;i<n;i++){
            freqs[s[right]]+=1;
            
            while(check(freqs,freqt)){
                cout<<left<<" "<<right<<endl;
                if(maxi>=right-left+1){
                    maxi=right-left+1;
                    min_left=left;
                    min_right=right;
                }
                freqs[s[left]]-=1;
                left+=1;
            }
            right++;
        }
        
        delete [] freqs;
        delete [] freqt;
        
        for(int i=min_left;i<=min_right;i++){
            ans+=s[i];
        }
        return ans;
        
        
    }

///////////////////////////////////////////////////////////////////////////////
///
///				Copy List with Random Pointer
////////////////////
  class Solution {
public:
    Node* copyRandomList(Node* head) {
        unordered_map<Node *,Node *> dp;
        Node* cur=head;
        
        while(cur!=NULL){
            dp[cur]=new Node(cur->val);
            cur=cur->next;
        }
        
        cur=head;
        while(cur!=NULL){
            dp[cur]->next=dp[cur->next];
            dp[cur]->random=dp[cur->random];
            cur=cur->next;
        }
        
        
        return dp[head];
        
    }
};

///////////////////////////////////////////////////////////////////////////////
///
//
///////////////////                         LRU  least recently used
///
///			o(1)get  and o(1)put 		map and doubly linkedlist

	class LRUCache{
	    size_t m_capacity;
	    unordered_map<int,  list<pair<int, int>>::iterator> m_map; //m_map_iter->first: key, m_map_iter->second: list iterator;
	    list<pair<int, int>> m_list;                               //m_list_iter->first: key, m_list_iter->second: value;
	public:
	    LRUCache(size_t capacity):m_capacity(capacity) {
	    }
	    int get(int key) {
	        auto found_iter = m_map.find(key);
	        if (found_iter == m_map.end()) //key doesn't exist
	            return -1;
	        m_list.splice(m_list.begin(), m_list, found_iter->second); //move the node corresponding to key to front
	        return found_iter->second->second;                         //return value of the node
	    }
	    void set(int key, int value) {
	        auto found_iter = m_map.find(key);
	        if (found_iter != m_map.end()) //key exists
	        {
	            m_list.splice(m_list.begin(), m_list, found_iter->second); //move the node corresponding to key to front
	            found_iter->second->second = value;                        //update value of the node
	            return;
	        }
	        if (m_map.size() == m_capacity) //reached capacity
	        {
	           int key_to_del = m_list.back().first; 
	           m_list.pop_back();            //remove node in list;
	           m_map.erase(key_to_del);      //remove key in map
	        }
	        m_list.emplace_front(key, value);  //create new node in list
	        m_map[key] = m_list.begin();       //create correspondence between key and node
	    }
	};


///////////////////////////////////////////////////////////////////////////////
///
///
///
///					////////  revserve a linked list
///					
///					
		 ListNode* reverseList(ListNode* head) {
        ListNode* prev=NULL,*temp;
        ListNode* cur=head;
        
        while(cur!=NULL){
            temp=cur->next;
            cur->next=prev;
            prev=cur;
            cur=temp;
        }
        
        return prev;
    }


///////////////////////////////////////////////////////////////////////////////
///
///						middle element in linkedlist
///
    	ListNode* middleNode(ListNode* head) {
        if(head==NULL){
            return head;
        }
        ListNode* cur=head, *nextp=head;
        
        while(nextp!=NULL and nextp->next!=NULL){
            cur=cur->next;
            nextp=nextp->next->next;
        }
        
        return cur;
        
    }


///////////////////////////////////////////////////////////////////////////////
///
///
///         ////////  978. Longest Turbulent Subarray
/// That is, the subarray is turbulent if the comparison sign flips between each 
/// adjacent pair of elements in the subarray.        

     int maxTurbulenceSize(vector<int>& arr) {
            int n=arr.size();
            
            int *dp=new int[n];
            int *sign=new int[n];
            
            
            int ans=0;
            
            for(int i=0;i<n;i++){
                dp[i]=1;
                sign[i]=0;
            }
            
            
            for(int i=1;i<n;i++){
                if(arr[i]<arr[i-1] and  (sign[i-1]==0 or sign[i-1]==-1)){
                    dp[i]+=dp[i-1];
                    sign[i]=1;
                }else if(arr[i]>arr[i-1] and  (sign[i-1]==0 or sign[i-1]==1)){
                    dp[i]+=dp[i-1];
                    sign[i]=-1;
                }
                else if(arr[i]<arr[i-1]) {
                    dp[i]+=1;
                    sign[i]=1;
                }else if(arr[i]>arr[i-1]){
                    dp[i]+=1;
                    sign[i]=-1;
                }else{
                    dp[i]=1;
                    sign[i]=0;
                }
            }
            
            
            for(int i=0;i<n;i++){
                //cout<<dp[i]<<" ";
                ans=max(ans,dp[i]);
            }
           // cout<<endl;
            // for(int i=0;i<n;i++){
            //     cout<<sign[i]<<" ";
            // }
            // cout<<endl;
            
            delete [] dp;
            delete [] sign;
            return ans;
        }

///////////////////////////////////////////////////////////////////////////////
///
///
//////////////       postorder and inorder to build a tree

class Solution {
public:
    int index=0;
    
    int find(int l,int r,vector<int>& inorder,int val){
        for(int i=l;i<=r;i++){
            if(inorder[i]==val){
                return i;
            }
        }
        return -1;
    }
    
    
    TreeNode *construct(vector<int>& postorder,vector<int>& inorder,int l,int r){
        if(l>r){
            return NULL;
        }
        
        TreeNode *root=new TreeNode(postorder[index++]);
        int pos=find(l,r,inorder,root->val);
        root->right=construct(postorder,inorder,pos+1,r);
        root->left=construct(postorder,inorder,l,pos-1);
        
        return root;   
    }
    
    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
        TreeNode* root=NULL;
        reverse(postorder.begin(),postorder.end());
        return construct(postorder,inorder,0,postorder.size()-1);
    }
};    


///////////////////////////////////////////////////////////////////////////////
///
///
//////////////       preorder and inorder to build a tree


class Solution {
public:
    int index=0;
    
    int find(int l,int r,vector<int>& inorder,int val){
        for(int i=l;i<=r;i++){
            if(inorder[i]==val){
                return i;
            }
        }
        
        return -1;
    }
    
    
    TreeNode *construct(vector<int>& p, vector<int>& i,int l,int r){
        if(l>r){
            return NULL;
        }
        
        TreeNode *root=new TreeNode(p[index++]);
        int pos=find(l,r,i,root->val);
        root->left=construct(p,i,l,pos-1);
        root->right=construct(p,i,pos+1,r);
        
        return root;
        
        
    }
    
    
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        return construct(preorder,inorder,0,preorder.size()-1);    
    }
};






///////////////////////////////////////////////////////////////////////////////
///
///                 equal partition to two parts
///                 
        bool solution(vector<int>& nums,int n,int goal,vector<vector<int>> &dp,int temp=0,int pos=0){
        if(temp==goal){
            return true;
        }
        
        if(pos>=n){
            return false;
        }
        
        if(dp[pos][temp]!=0){
            if(dp[pos][temp]==1) return true;
            else return false;
        }
        
        if(solution(nums,n,goal,dp,temp+nums[pos],pos+1)  or solution(nums,n,goal,dp,temp,pos+1) ){
            dp[pos][temp]=1;
        }
        else{
            dp[pos][temp]=2;
        }
        
        if(dp[pos][temp]==1){
            return true;
        }
        return false;
        
         
    }
    

    bool canPartition(vector<int>& nums) {
        int n=nums.size();
        int sum_=0;
        for(int i:nums){
            sum_+=i;
        }
        
        if(sum_%2!=0){
            return false;
        }
        
        vector<vector<int>> dp(n+1,vector<int>(sum_+1,0));
        // 0 for not visted 
        // 1 for true
        // 2 for false
        
        return solution(nums,n,sum_/2,dp);
        
        
    }
///////////////////////////////////////////////////////////////////////////////
///
///
////////            Partition to K Equal Sum Subsets


        ////// backtracking approach
    
     int n;
    bool canPartitionKSubsets(vector<int>& nums, int k) {
        
        n=nums.size();
        int sum_=0;
        for(int i:nums){
            sum_+=i;
        }
        
        if(k==0 or k>n or sum_%k!=0){
            return false;
        }
        
        vector<bool> visted(n,false);
        
        return solution(nums,k,visted,sum_/k);
    }
    
    bool solution(vector<int>& nums, int k, vector<bool> &visted,int target_sum,int temp_sum=0,int start=0){
        if(k==1) return true;
        
        if(temp_sum==target_sum){
            return solution(nums,k-1,visted,target_sum);
        }
        
        for(int i=start;i<n;i++){
            if(!visted[i]){
                visted[i]=true;
                if(solution(nums,k,visted,target_sum,temp_sum+nums[i],i+1)) return true;
                visted[i]=false;
            }
        }
        return false;
    }



///////////////////////////////////////////////////////////////////////////////
///
///////                     940. Distinct Subsequences II   (HARD)
///
///
    class Solution {
    public:
    int mod=1e9+7;
    int distinctSubseqII(string s) {
        int n=s.size();
        s="#"+s;
        vector<int> dp(n+1,0);
        unordered_map<char,int> postion;
        dp[0]=1;
        for(int i=1;i<=n;i++){
            dp[i]=dp[i-1]*2%mod;
            if(postion.find(s[i])!=postion.end()){
                dp[i]=(dp[i]+mod-dp[postion[s[i]]-1])%mod;
            }
            
            postion[s[i]]=i;
        }
        return dp[n]-1;
    }
};
///////////////////////////////////////////////////////////////////////////////
///
//////////          873. Length of Longest Fibonacci Subsequence

    class Solution {
        public:
          int lenLongestFibSubseq(vector<int>& A) {
            const int n = A.size();    
            unordered_set<int> m(begin(A), end(A));    
            int ans = 0;
            for (int i = 0; i < n; ++i)
              for (int j = i + 1; j < n; ++j) {
                int a = A[i];
                int b = A[j];
                int c = a + b;
                int l = 2;
                while (m.count(c)) {
                  a = b;
                  b = c;
                  c = a + b;
                  ans = max(ans, ++l);
                }
              }
            return ans;
          }
        };



///////////////////////////////////////////////////////////////////////////////
///
/////////////               knapsack 0/1
///


        int *weight=new int[n];
        int *val=new int[n];
        
        for(int i=0;i<n;i++){
            cin>>val[i];
        }
        for(int i=0;i<n;i++){
            cin>>weight[i];
        }
        
        int **dp=new int*[n+1];
        for(int i=0;i<n+1;i++){
            dp[i]=new int[w+1];
        }     
            
        
        for(int i=0;i<n+1;i++){
            for(int j=0;j<w+1;j++){
                if(i==0 or j==0){
                    dp[i][j]=0;
                    continue;
                }
                
                if(weight[i-1]<=j){
                    dp[i][j]=max(dp[i-1][j], val[i-1] + dp[i-1][j-weight[i-1]]);
                }
                else{
                    dp[i][j]=dp[i-1][j];
                }
            }
        }    
        cout<<dp[n][w]<<endl;
        
        
        delete [] weight;
        delete [] val;


///////////////////////////////////////////////////////////////////////////////    
///
///////////   Count Number of Binary Tree Possible given Preorder Sequence              
///

        /// USING CATALYN NUMBER
        

    ans[0]=1;
    ans[1]=1;

    for(ll i=2;i<n;i++){
        for(ll j=0;j<i;j++){
            ans[i]+=ans[j] * ans[i-j-1];
        }
    }


    /// DIRECT FORMULA

    (2n)!/((n+1)! * (n)!)

        

///////////////////////////////////////////////////////////////////////////////
///
//////////          Numbers WIthout Consecutive 1s in binary representation
///

//        if n==1 : possible test case is 0 and 1 ans is --> 2
//        if n==2 : possible test case is 00 and 01 and 10 ans is --> 3
//        if n==3 : possible test case is 000 001 010 100 101 is --> 5


Fibonacci with f(n+1)           /// pay attention to n+1

///////////////////////////////////////////////////////////////////////////////
///
///////             coin changing in min number of steps 
///
        class Solution {
        public:
            int big_number=10000000;
            int coinChange(vector<int>& coins, int amount) {
                int n=coins.size();
                int ans=-1;
                
                int *dp=new int[amount+1];
                for(int i=0;i<amount+1;i++){
                    if(i==0){
                        dp[i]=0;
                        continue;
                    }
                    dp[i]=big_number;
                }
                
                for(int i=0;i<n;i++){
                    for(int j=0;j<amount+1;j++){
                        if(j<coins[i]){
                            continue;
                        }
                        dp[j]=min(dp[j],1+dp[j-coins[i]]);
                        
                    }
                }
                   
                if(dp[amount]!=big_number)
                    ans=dp[amount];
                    
                delete [] dp;
                
                return ans;
                
            }
        };

/////////////////               Coin Change total number of ways to get total
///


        signed main(){
            ios_base::sync_with_stdio(false);
            cin.tie(nullptr);
            
            ll test;
            cin>>test;
            while(test--){
                ll n;
                cin>>n;
                ll *coins=new ll[n];
                for(ll i=0;i<n;i++){
                    cin>>coins[i];
                }
                
                ll k;
                cin>>k;
                
                ll **dp=new ll*[n+1];
                for(ll i=0;i<n+1;i++){
                    dp[i]=new ll[k+1];
                }
                
                for(ll i=0;i<n+1;i++){
                    for(ll j=0;j<k+1;j++){
                        if(i==0 or j==0) {
                            dp[i][j]=0;
                            continue;
                            
                        }                
                        if(coins[i-1]>j) {
                            dp[i][j]=dp[i-1][j];
                            continue;
                        }                
                        if(coins[i-1]==j) dp[i][j]=1+dp[i-1][j];
                        else{
                            dp[i][j]=dp[i-1][j]+dp[i][j-coins[i-1]];
                        }
                        
                    }
                }
                
                // for(ll i=0;i<n+1;i++){
                //     for(ll j=0;j<k+1;j++){
                //       cout<<dp[i][j]<<" ";
                //     }
                //     br;
                // }
                
                cout<<dp[n][k];br;
                
                for(ll i=0;i<n+1;i++){
                    delete [] dp[i];
                }
                
                delete [] dp;
                
                delete [] coins;
                
                
            }

            
        }



///////////////////////////////////////////////////////////////////////////////
///
////////            Longest Bitonic Subsequence
///


        void solve(vector<ll> arr,vector<ll> &dp){       // longest increasing subseq
            ll len=arr.size();
            for(ll i=1;i<len;i++){
                for(ll j=0;j<i;j++){
                    if(arr[i]>arr[j]){
                        dp[i]=max(dp[i],dp[j]+1);
                    }
                }
            }
        }
        signed main() {
            ios_base::sync_with_stdio(false);
            cin.tie(nullptr);
            
            ll test;
            cin>>test;
            while(test--){
                
                vector<ll> inc,dec,arr;
                
                ll n;
                cin>>n;
                
                arr.resize(n);
                inc.assign(n,1);
                dec.assign(n,1);
                
                for(ll i=0;i<n;i++){
                    cin>>arr[i];
                }
                solve(arr,inc);
                reverse(arr.begin(),arr.end());
                solve(arr,dec);  
                reverse(dec.begin(),dec.end());                    
                ll ans=0;
                for(ll i=0;i<n;i++){
                    ans=max(ans,inc[i]+dec[i]-1);
                }
                
                cout<<ans;br;
                
            }
            return 0;
        }



///////////////////////////////////////////////////////////////////////////////
///
//////////////          Total Ways in Matrix 

    vector<vector<int>> grid;
    
    int solve(int row,int col,int x=0,int y=0){
        if(x>=row or y>=col){
            return 0;
        }
        
        if(x==row-1 and y==col-1){
            return 1;
        }
        
        if(grid[x][y]!=0){
            return grid[x][y];
        }
        
        int right=solve(row,col,x,y+1);
        int down=solve(row,col,x+1,y);
        
        grid[x][y]=right+down;
        return right+down;
        
    }
    
    int uniquePaths(int m, int n) {
        grid.assign(m+1,vector<int>(n+1,0));
        return solve(m,n);
    }


////////            best way to reach end with obstacles

    class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        if obstacleGrid[0][0] == 1:
            return 0
        obstacleGrid[0][0] = 1
        for i in range(1,m):
            obstacleGrid[i][0] = int(obstacleGrid[i][0] == 0 and obstacleGrid[i-1][0] == 1)  
        for j in range(1, n):
            obstacleGrid[0][j] = int(obstacleGrid[0][j] == 0 and obstacleGrid[0][j-1] == 1)
        for i in range(1,m):
            for j in range(1,n):
                if obstacleGrid[i][j] == 0:
                    obstacleGrid[i][j] = obstacleGrid[i-1][j] + obstacleGrid[i][j-1]
                else:
                    obstacleGrid[i][j] = 0      
        return obstacleGrid[m-1][n-1]

///////////////////////////////////////////////////////////////////////////////
///
////////        Count Number of Binary Search Tree Possible given n 
///
///

       // same as CATALYN number

///////////////////////////////////////////////////////////////////////////////
///
///
////////        String Interleaving (HARD)

        class Solution {
public:
    int n1,n2,n3;
    
    bool isInterleave(string s1, string s2, string s3) {
        n1=s1.size();
        n2=s2.size();
        n3=s3.size();
        
        ///.  dp solution
        
        if(n1+n2!=n3){
            return false;
        }
        
        vector<vector<bool>> dp(n1+1,vector<bool>(n2+1,false));
        
        for(int i=0;i<=n1;i++){
            for(int j=0;j<=n2;j++){
                int l = i + j -1;
                if(i==0 and j==0){
                    dp[i][j]=true;
                    continue;
                }
                if(i==0){
                    if(s2[j-1]==s3[l]){
                        dp[i][j]=dp[i][j-1];   
                    }
                    continue;
                }else if(j==0){
                    if(s1[i-1]==s3[l]){
                        dp[i][j]=dp[i-1][j];
                    }
                    continue;
                }
                
                dp[i][j]=(s1[i-1]==s3[l]?dp[i-1][j]: false )  or (s2[j-1]==s3[l]?dp[i][j-1]: false ); 
                
            }
        }
        
        return dp[n1][n2];
        
    }
};


///////////////////////////////////////////////////////////////////////////////
///
///////             Maximum Sum Increasing Subsequence
///
                
        signed main()
        {
            ios_base::sync_with_stdio(false);
            cin.tie(nullptr);
            
            ll test;
            cin>>test;
            while(test--){
                ll n;
                cin>>n;
                ll *arr=new ll[n];
                ll *dp=new ll[n];
                ll _max=0;
                
                for(ll i=0;i<n;i++){
                    cin>>arr[i];
                    dp[i]=arr[i];
                    _max=max(_max,dp[i]);
                }
                
                
                
                for(ll i=1;i<n;i++){
                    for(ll j=0;j<i;j++){
                        if(arr[i]>arr[j]){
                            dp[i]=max(dp[i],dp[j]+arr[i]);
                            _max=max(_max,dp[i]);
                        }
                    }
                }
                
                cout<<_max;br;
                delete [] dp;
                delete [] arr;
            }
            
        }
/////////////////////////////////////////////////////////////////////////////////
///
///////             print sprial of matrix
///

         vector<int> spiralOrder(vector<vector<int>>& matrix) {
        int row,col;
        vector<int> ans;
        row=matrix.size();
        if(row==0){
            return ans;
        }
        
        col=matrix[0].size();
        
        int start_row=0,start_col=0,end_row=row-1,end_col=col-1;
        
        while(start_row<=end_row and start_col<=end_col){
            // column horizontal
            
            for(int i=start_col;i<=end_col;i++){
                ans.push_back(matrix[start_row][i]);
            }
            start_row+=1;
            
            // row vertical
            
            for(int i=start_row;i<=end_row;i++){
                ans.push_back(matrix[i][end_col]);
            }
            end_col-=1;
            
            if(start_row<=end_row){
                for(int i=end_col;i>=start_col;i--){
                    ans.push_back(matrix[end_row][i]);
                }
                end_row-=1;
            }
            
            if(start_col <= end_col){
                for(int i=end_row;i>=start_row;i--){
                    ans.push_back(matrix[i][start_col]);
                }
                start_col+=1;
            }
            
        }
        
        return ans;
    }


/////////////////////////////////////////////////////////////////////////////////
///
///
//////////          Word Boggle

        #include<bits/stdc++.h> 
        using namespace std; 

        #define char_int(c) ((int)c - (int)'A') 


        #define SIZE (26) 

        #define M 3 
        #define N 3 

        struct TrieNode 
        { 
            TrieNode *Child[SIZE]; 
            bool leaf; 
        }; 

        TrieNode *getNode() 
        { 
            TrieNode * newNode = new TrieNode; 
            newNode->leaf = false; 
            for (int i =0 ; i< SIZE ; i++) 
                newNode->Child[i] = NULL; 
            return newNode; 
        } 


        void insert(TrieNode *root, char *Key) 
        { 
            int n = strlen(Key); 
            TrieNode * pChild = root; 

            for (int i=0; i<n; i++) 
            { 
                int index = char_int(Key[i]); 

                if (pChild->Child[index] == NULL) 
                    pChild->Child[index] = getNode(); 

                pChild = pChild->Child[index]; 
            } 


            pChild->leaf = true; 
        } 


        bool isSafe(int i, int j, bool visited[M][N]) 
        { 
            return (i >=0 && i < M && j >=0 && 
                    j < N && !visited[i][j]); 
        } 

        void searchWord(TrieNode *root, char boggle[M][N], int i, 
                        int j, bool visited[][N], string str) 
        { 
            if (root->leaf == true) 
                cout << str << endl ; 
            if (isSafe(i, j, visited)) 
            { 
                visited[i][j] = true; 
                for (int K =0; K < SIZE; K++) 
                { 
                    if (root->Child[K] != NULL) 
                    { 
                        char ch = (char)K + (char)'A' ; 
                        if (isSafe(i+1,j+1,visited) && boggle[i+1][j+1] == ch) 
                            searchWord(root->Child[K],boggle,i+1,j+1,visited,str+ch); 
                        if (isSafe(i, j+1,visited) && boggle[i][j+1] == ch) 
                            searchWord(root->Child[K],boggle,i, j+1,visited,str+ch); 
                        if (isSafe(i-1,j+1,visited) && boggle[i-1][j+1] == ch) 
                            searchWord(root->Child[K],boggle,i-1, j+1,visited,str+ch); 
                        if (isSafe(i+1,j, visited) && boggle[i+1][j] == ch) 
                            searchWord(root->Child[K],boggle,i+1, j,visited,str+ch); 
                        if (isSafe(i+1,j-1,visited) && boggle[i+1][j-1] == ch) 
                            searchWord(root->Child[K],boggle,i+1, j-1,visited,str+ch); 
                        if (isSafe(i, j-1,visited)&& boggle[i][j-1] == ch) 
                            searchWord(root->Child[K],boggle,i,j-1,visited,str+ch); 
                        if (isSafe(i-1,j-1,visited) && boggle[i-1][j-1] == ch) 
                            searchWord(root->Child[K],boggle,i-1, j-1,visited,str+ch); 
                        if (isSafe(i-1, j,visited) && boggle[i-1][j] == ch) 
                            searchWord(root->Child[K],boggle,i-1, j, visited,str+ch); 
                    } 
                } 

                visited[i][j] = false; 
            } 
        } 

        void findWords(char boggle[M][N], TrieNode *root) 
        { 
            bool visited[M][N]; 
            memset(visited,false,sizeof(visited)); 

            TrieNode *pChild = root ; 

            string str = ""; 
            for (int i = 0 ; i < M; i++) 
            { 
                for (int j = 0 ; j < N ; j++) 
                { 
                    if (pChild->Child[char_int(boggle[i][j])] ) 
                    { 
                        str = str+boggle[i][j]; 
                        searchWord(pChild->Child[char_int(boggle[i][j])], 
                                boggle, i, j, visited, str); 
                        str = ""; 
                    } 
                } 
            } 
        } 

        int main() 
        { 
            char *dictionary[] = {"GEEKS", "FOR", "QUIZ", "GEE"}; 
            TrieNode *root = getNode(); 
            int n = sizeof(dictionary)/sizeof(dictionary[0]); 
            for (int i=0; i<n; i++) 
                insert(root, dictionary[i]); 

            char boggle[M][N] = {{'G','I','Z'}, 
                {'U','E','K'}, 
                {'Q','S','E'} 
            }; 

            findWords(boggle, root); 

            return 0; 
        } 
// https://practice.geeksforgeeks.org/problems/word-boggle/0  (do this)

/////////////////////////////////////////////////////////////////////////////////
///
///////                  merge sort
///




/////////////////////////////////////////////////////////////////////////////////
///
///
////////                quick sort



/////////////////////////////////////////////////////////////////////////////////
///
/////////                       top view of tree
///

#include<bits/stdc++.h> 
using namespace std; 

struct Node{ 
    Node * left; 
    Node* right; 
    int data; 
}; 


Node* newNode(int key){ 
    Node* node=new Node(); 
    node->left = node->right = NULL; 
    node->data=key; 
    return node; 
} 

void fillMap(Node* root,int d,int l,map<int,pair<int,int>> &m){ 
    if(root==NULL) return; 

    if(m.count(d)==0){ 
        m[d] = make_pair(root->data,l); 
    }else if(m[d].second>l){ 
        m[d] = make_pair(root->data,l); 
    } 

    fillMap(root->left,d-1,l+1,m); 
    fillMap(root->right,d+1,l+1,m); 
} 

void topView(struct Node *root){ 
    map<int,pair<int,int>> m; 
    fillMap(root,0,0,m); 

    for(auto it=m.begin();it!=m.end();it++){ 
        cout << it->second.first << " "; 
    } 
} 

int main(){ 
    Node* root = newNode(1); 
    root->left = newNode(2); 
    root->right = newNode(3); 
    root->left->right = newNode(4); 
    root->left->right->right = newNode(5); 
    root->left->right->right->right = newNode(6); 
    cout<<"Following are nodes in top view of Binary Tree\n"; 
    topView(root); 
    return 0; 
} 

/////////////////////////////////////////////////////////////////////////////////
///
/////////////  vertical level order traversal
///

   class Solution {
public:
    void helper(TreeNode* root, int temp,int height,map<int,map<int,set<int>>> &m){
        if(root==NULL){
            return;
        }
        m[temp][height].insert(root->val);
        helper(root->left,temp-1,height+1,m);
        helper(root->right,temp+1,height+1,m);
    }
    
    vector<vector<int>> verticalTraversal(TreeNode* root) {
        map<int,map<int,set<int>>> m;
        helper(root,0,0,m);
        
        vector<vector<int>> sol;
        vector<int> ans;
        auto ptr=m.begin();
        while(ptr!=m.end()){
            auto ptr2=ptr->second.begin();
            ans.clear();
            while(ptr2!=ptr->second.end()){
                set<int> temp=ptr2->second;
                auto i=temp.begin();
                while(i!=temp.end()){
                    ans.push_back(*i);
                    i++;
                }
                ptr2++;
            }
            sol.push_back(ans);
            
            ptr++;
        }
        return sol;
        
    }
};
/////////////////////////////////////////////////////////////////////////////////
///
/////////               bottom view



void printBottomViewUtil(Node * root, int curr, int hd, map <int, pair <int, int>> & m) 
{ 
    if (root == NULL) 
        return; 
    if (m.find(hd) == m.end()) 
    { 
        m[hd] = make_pair(root -> data, curr); 
    } 
    else
    { 
        pair < int, int > p = m[hd]; 
        if (p.second <= curr) 
        { 
            m[hd].second = curr; 
            m[hd].first = root -> data; 
        } 
    } 
    printBottomViewUtil(root -> left, curr + 1, hd - 1, m); 
    printBottomViewUtil(root -> right, curr + 1, hd + 1, m); 
} 




void bottomView(Node *root)
{
   map < int, pair < int, int > > m; 
    printBottomViewUtil(root, 0, 0, m); 
    map < int, pair < int, int > > ::iterator it; 
    for (it = m.begin(); it != m.end(); ++it) 
    { 
        pair < int, int > p = it -> second; 
        cout << p.first << " "; 
    } 
}


/////////////////////////////////////////////////////////////////////////////////
///
///////     MATRIX CHAIN MULTIPLICATION
///


        ////   RECURSIVE SOLUTION (SHOWED TLE)

        unordered_map<ll,unordered_map<ll,ll>> store;


        ll matrix(ll *dp,ll start,ll end){
            
            if(store[start][end]!=0)
            {
                
                return store[start][end];
            
            }
            
            if(start==end) return 0;
            
            
            ll min_=INT_MAX;
            
            ll count;
            
            for(ll k=start;k<end;k++){
                
                count=matrix(dp,start,k)+matrix(dp,k+1,end)+dp[k]*dp[start-1]*dp[end];
                
                min_=min(min_,count);
            
            }
            
            store[start][end]=min_;
            
            return min_;
        }


        signed main(){
            ios_base::sync_with_stdio(false);
            cin.tie(nullptr);
            
            ll test;
            cin>>test;
            while(test--){
                
                store.clear();
                
                ll n;
                cin>>n;
                
                ll *dp=new ll[n];
                for(ll i=0;i<n;i++){
                    cin>>dp[i];
                }
                
                cout<<matrix(dp,1,n-1);
                
                delete [] dp;
                
                br;
            }
        }


        ///    DYNAMIC BOTTOM UP SOLUTION 
        

        typedef long long ll;

        #define br cout<<"\n"

        signed main(){
            ios_base::sync_with_stdio(false);
            cin.tie(nullptr);
            
            ll test;
            cin>>test;
            while(test--){
                ll n;
                cin>>n;
                ll *arr=new ll[n];
                ll **dp=new ll*[n];
                
                for(ll i=0;i<n;i++){
                    cin>>arr[i];
                    dp[i]=new ll[n];
                    for(ll j=0;j<n;j++){
                        dp[i][j]=0;
                    }
                }
                
                for(ll l=2;l<n;l++){
                    for(ll i=0;i<n-l;i++){
                        ll j=i+l;
                        dp[i][j]=INT_MAX;
                        for(ll k=i+1;k<j;k++){
                            dp[i][j]=min(dp[i][j],dp[i][k]+dp[k][j]+arr[i]*arr[k]*arr[j]);
                        }
                    }
                }
                
                cout<<dp[0][n-1];br;
                
                for(ll i=0;i<n;i++){
                    delete [] dp[i];
                }
                
                delete [] dp;
                
                delete [] arr;
            }
            
        }

/////////////////////////////////////////////////////////////////////////////////
///
///////                 Minimum Edit Distance            
///

        class Solution {
public:
    int minDistance(string word1, string word2) {
        
        int l1=word1.size();            // row
        int l2=word2.size();            // col
        
        vector<vector<int>> dp(l1+1,vector<int>(l2+1));
        
        
        for(int i=0;i<l1+1;i++){
            dp[i][0]=i;    
        }
        
        for(int i=0;i<l2+1;i++){
            dp[0][i]=i;
        }
        
        for(int i=1;i<l1+1;i++){
            for(int j=1;j<l2+1;j++){
                if(word1[i-1]!=word2[j-1]){
                    dp[i][j]=min({dp[i-1][j],dp[i][j-1],dp[i-1][j-1]})+1;
                }
                else{
                    dp[i][j]=dp[i-1][j-1];
                }
            }
        }
        
        return dp[l1][l2];
        
    }
};


/////////////////////////////////////////////////////////////////////////////////
///
////////            Optimal Binary Search Tree 
///

    
ll sum_range(ll *arr,ll start,ll end){
    ll sum=0;
    for(ll i=start;i<=end;i++){
        sum+=arr[i];
    }
    return sum;
}

ll solve(ll *arr,ll *freq,ll n){
    ll **dp=new ll*[n];
    ll ans=0;
    for(ll i=0;i<n;i++){
        dp[i]=new ll[n];
        for(ll j=0;j<n;j++){
            dp[i][j]=0;
        }
    }

    for(ll i=0;i<n;i++){
        dp[i][i]=freq[i];
    }

    ll temp;
    for(ll l=2;l<=n;l++){
        for(ll i=0;i<=n-l;i++){
            ll j=i+l-1;
            dp[i][j]=INT_MAX;
            ll sum=sum_range(freq,i,j);
            for(ll k=i;k<=j;k++){
                temp=sum+(k-1<i ? 0 : dp[i][k-1])+ (k+1>j? 0: dp[k+1][j]);
                dp[i][j]=min(dp[i][j],temp);
            }
        }
    }

    for(ll i=0;i<n;i++){
        for(ll j=0;j<n;j++){
            cout<<dp[i][j]<<" ";
        }
        br;
    }

    ans=dp[0][n-1];


    for(ll i=0;i<n;i++){
        delete [] dp[i];
    }

    delete [] dp;

    return ans;
}
    



/////////////////////////////////////////////////////////////////////////////////
///
///////////                 114. Flatten Binary Tree to Linked List
///



/////////////////////////////////////////////////////////////////////////////////
///
///
//////////                  Cutting Rod   Maximize The Cut 

        ll test;
    cin>>test;
    while(test--){
        ll n;
        cin>>n;
        ll x[3];
        for(ll i=0;i<3;i++){
            cin>>x[i];
        }
        
        ll *dp=new ll[n+1];
        
        for(ll i=0;i<n+1;i++){
            dp[i]=neg_inf;
        }
        dp[0]=0;
        
        for(ll i=0;i<3;i++){
            for(ll j=1;j<n+1;j++){
                if(x[i]>j){
                    continue;
                }        
                dp[j]=max(dp[j],1+dp[j-x[i]]);
            
            }
            // for(ll i=0;i<n+1;i++){
            //     cout<<dp[i]<<" ";
            // }
            // br;
        }
                
        
        cout<<dp[n];br;
        
        delete [] dp;


/////////////////////////////////////////////////////////////////////////////////
///
///////////                 wildcard pattern
///







/////////////////////////////////////////////////////////////////////////////////
///
/////////               Regular Expression Dynamic Programming
///

class Solution {
public:
    bool check(string s,string p,int sl,int pl){
        if(pl==-1) return sl==-1;         
        
        if(p[pl]=='*'){               
            if(check(s,p,sl,pl-2)){
                return true;            //  # s = ba, p = bab*
            }
            if(sl>=0 and (s[sl]==p[pl-1] or p[pl-1]=='.') and check(s,p,sl-1,pl)){
                return true;          //   # s = ba, p = ba* or b.*
            }
        }
        
        if(sl>=0. and (s[sl]==p[pl] or p[pl]=='.') and check(s,p,sl-1,pl-1)){
            return true;        // # s = bac, p = bac or ba.
        }
        
        return false;
    }

    bool isMatch(string s, string p) {
        int sl=s.size();
        int pl=p.size();
        
        return check(s,p,sl-1,pl-1);
        
    }
};


/////////////////////////////////////////////////////////////////////////////////
///
///////             max area in histogram 
                    zigzag pattern                    
                    binary tree to doubly linked list


/////////////////////////////////////////////////////////////////////////////////
///
/////////           word break problem

        int n;
         cin>>n;
         string s;
         unordered_map<string,int> m;
         for(int i=0;i<n;i++){
             cin>>s;
             m[s]++;
         }
         cin>>s;
         n=s.length();
         vector<bool> dp(n+1,false);
         dp[0]=true;
         string t1,t="";
         for(int i=1;i<=n;i++){
             t+=s[i-1];
             t1="";
             for(int j=i-1;j>=0;j--){
                t1=s[j]+t1;
                if(m.find(t1)!=m.end() && dp[j]){
                    dp[i]=true;
                } 
             }
         }
         
         if(dp[n]){
             cout<<"1"<<endl;
         }
         else{
             cout<<"0"<<endl;
         }



/////////////////////////////////////////////////////////////////////////////////
///
///    26. Remove Duplicates from Sorted Array


static auto x = []() {ios_base::sync_with_stdio(false); cin.tie(NULL); return NULL; }();

class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        short int i(0);
        if(nums.size()==0) { return 0; }
        for(short int j(0);j<nums.size();j++)
        {
            if(nums[i]==nums[j]) { continue; }
            else { nums[++i]=nums[j]; }
        }
        return i+1;      
    }
};

/////////////      my bad solution

    class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        string positive="",negative="";
        int max_val=10000;
        for(int i=0;i<max_val;i++){
            positive+='0';
            negative+='0';
        }
        
        int n=nums.size();
        int ind=0;
        for(int i=0;i<n;i++){
            if(nums[ind]>=0){
                if(positive[nums[ind]]=='1'){
                    nums.erase(nums.begin()+ind);
                    continue;
                }

                positive[nums[ind]]='1';
                ind+=1;
            }else{
                
                if(negative[-nums[ind]]=='1'){
                    nums.erase(nums.begin()+ind);
                    continue;
                }

                negative[-nums[ind]]='1';
                ind+=1;
                
            } 
        }
    
        
        
        int ans=0;
        for(int i=0;i<max_val;i++){
            if(positive[i]=='1'){
                ans+=1;
            }
            if(negative[i]=='1'){
                ans+=1;
            }
        }
        
        return ans;
    }
};




///////////////////////////////////////////////   Remove Duplicates from Sorted Array II
///
    [1,1,1,2,2,3]   --->>     [1,1,2,2,3]  at max 2 value allowed

static auto fast_input=[] (){ios_base::sync_with_stdio(false); cin.tie(nullptr); 
    return NULL;}();

class Solution {
public:
    int removeDuplicates(vector<int>& A) {
        int n=A.size();
        if(n == 0){
            return 0;
        }
        int p = 0;
        int dupTimes = 0;
        for(int i = 1; i < n; i++){
            if(A[i] != A[p]){
                A[++p] = A[i];
                dupTimes = 0;
            }
            else{
                dupTimes++;
                if(dupTimes == 1){
                    A[++p] = A[i];
                }
            }
        }
        return p + 1;
    }
};



/////////////////////////////////////////////////////////////////////////////////
///
///////                 best day to buy stocks 2 buy 1 sell 1 again buy 2 sell 2
///

    class Solution {
public: 
    // iterative solution
    // int maxProfit(vector<int>& pri) {
    //     int buy = 0, sell = 0, ans = 0, n = pri.size();
    //     while (buy < n && sell < n) {
    //         while (buy + 1 < n && pri[buy + 1] < pri[buy])
    //             buy++; 
    //         sell = buy; 
    //         while (sell + 1 < n && pri[sell + 1] > pri[sell])
    //             sell++;
    //         ans += pri[sell] - pri[buy];
    //         buy = sell + 1;
    //     }
    //     return ans;
    // }
    
    // dynamic programming
    
    int maxProfit(vector<int>& prices) {
          if(prices.size() <= 1) return 0;

          int res = 0;

          for( size_t i = 1; i < prices.size(); i++)
            if( prices[i] - prices[i-1] > 0 ) 
              res += prices[i] - prices[i-1];

          return res;
    }
};



/////////////////////////////////////////////////////////////////////////////////
///
///                             aggresive cows
    
int INF=1e9+7;
ll ans;
ll n,k;
vector<ll> arr;

bool check(ll gap){
    ll cows_left=k-1;
    ll last_pos=0;
    for(ll i=1;i<n;i++){
        if(cows_left==0){
            return true;
        }
        if(arr[i]-arr[last_pos]>=gap){
            cows_left-=1;
            last_pos=i;
        }
        if(cows_left==0){
            return true;
        }
    }
    return false;
}



signed main(){
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    ll test;
    cin>>test;
    while(test--){

        cin>>n>>k;
        ans=1;
        arr.resize(n);
        ll max_value=-INF;


        for(ll i=0;i<n;i++){
            cin>>arr[i];
            max_value=max(max_value,arr[i]);
        }

        sort(arr.begin(),arr.end());
        
        if(n==1){
            cout<<arr[0];br;
            continue;
        }
        
        ll start=0;
        ll end=arr[n-1]-arr[0];
        ll i;
        while(start<=end){
            ll mid=(start+end)/2;
            if(check(mid)){
                ans=mid;
                start=mid+1;
            }else{
                end=mid-1;
            }
        }

        cout<<ans;br;


    }
}


/////////////////////////////////////////////////////////////////////////////////
///
///////                     Inversion count

// using merge sort


ll ans;

void merge(vector<ll> &arr,ll start,ll end,ll mid){

    ll i=start;
    ll j=mid+1;
    ll k=0;
    vector<ll> new_arr(end-start+1);
    while(i<=mid and j<=end){
        if(arr[i]>arr[j]){
            new_arr[k++]=arr[j];
            j+=1;
            ans+=((mid+1)-i);           // mid+1 because j is starting from there
        }else{
            new_arr[k++]=arr[i];
            i+=1;
        }
    }

    while(i<=mid){
        new_arr[k++]=arr[i];
        i+=1;
    }

    while(j<=end){
        new_arr[k++]=arr[j];
        j+=1;
    }
    k=0;
    for(ll x=start;x<=end;++x,++k){
        arr[x]=new_arr[k];
    }

}


void mergesort(vector<ll> &arr,ll start,ll end){

    if(start>=end){
        return;
    }
    ll mid=(start+end)/2;
    mergesort(arr,start,mid);
    mergesort(arr,mid+1,end);
    merge(arr,start,end,mid);

}



signed main(){
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    ll n;
    cin>>n;
    vector<ll> arr(n);

    for(ll i=0;i<n;i++){
        cin>>arr[i];
    }
    ans=0;
    mergesort(arr,0,n-1);
    cout<<ans;
    br;

}

/////////////////////////////////////////////////////////////////////////////////
///
////////////                                diameter of tree


class Solution {
public:
    int ans=1;
    int depth(TreeNode* &root){
        if(root==NULL){
            return 0;
        }
        int left=depth(root->left);
        int right=depth(root->right);
        ans=max(ans,left+right+1);
        return max(left,right)+1;
        
    }
    int diameterOfBinaryTree(TreeNode* root) {
        depth(root);
        return ans-1;
    }
};


/////////////////////////////////////////////////////////////////////////////////
///
//////////                                  level order traversal

    void levelOrder(Node* node)
{
  queue<Node *> q;
  
  q.push(node);
  
  
  int this_level=1;
  int next_level=0;
  
  Node *up=NULL;
  
  while(!q.empty()){
      while(this_level!=0){
          up=q.front();
          q.pop();
          this_level-=1;
          if(up!=NULL){
            cout<<up->data<<" ";
            
            if(up->left!=NULL){
                q.push(up->left);
                next_level+=1;
            } 
            
            if(up->right!=NULL){
                q.push(up->right);
                next_level+=1;
            }    
          }
      }
      if(this_level==0){
         // cout<<"\n";
          this_level=next_level;
          next_level=0;
      }
  }
  
  
}

/////////////////////////////////////////////////////////////////////////////////
///
////////      level order spiral printing tree  and  height of tree
///


int height(struct Node* node) 
{ 
    if (node == NULL) 
        return 0; 
    else { 
        int lheight = height(node->left); 
        int rheight = height(node->right); 
        if (lheight > rheight) 
            return (lheight + 1); 
        else
            return (rheight + 1); 
    } 
} 


  
void printGivenLevel(struct Node* root, int level, int ltr) 
{ 
    if (root == NULL) 
        return; 
    if (level == 1) 
        printf("%d ", root->data); 
    else if (level > 1) { 
        if (ltr) { 
            printGivenLevel(root->left, level - 1, ltr); 
            printGivenLevel(root->right, level - 1, ltr); 
        } 
        else { 
            printGivenLevel(root->right, level - 1, ltr); 
            printGivenLevel(root->left, level - 1, ltr); 
        } 
    } 
} 



void printSpiral(struct Node* root) 
{ 
    int h = height(root); 
    int i; 
    bool ltr = false; 
    for (i = 1; i <= h; i++) { 
        printGivenLevel(root, i, ltr); 
        ltr = !ltr; 
    } 
} 
/////////////////////////////////////////////////////////////////////////////////
///
///
/////////                       left view of tree

class Solution {
public:
    vector<int> ans;
    void dfs(TreeNode* root,int &max_temp,int temp){
        if(root==NULL){
            return ;
        }
        if(temp>max_temp){
            ans.push_back(root->val);
            max_temp=temp;
        }
        
        dfs(root->left,max_temp,temp+1);
        dfs(root->right,max_temp,temp+1);
        
    }
    
    
    vector<int> rightSideView(TreeNode* root) {
        int max_temp=0, temp=1;
        dfs(root,max_temp,temp);
        
        return ans;
    }
    
    
    
};

/////////////////////////////////////////////////////////////////////////////////
///
//////////                      least common ansector  in graphs


int n, l;
vector<vector<int>> adj;

int timer;
vector<int> tin, tout;
vector<vector<int>> up;

void dfs(int v, int p)
{
    // cout<<"yo";br;
    tin[v] = ++timer;
    up[v][0] = p;
    for (int i = 1; i <= l; ++i)
        up[v][i] = up[up[v][i-1]][i-1];

    for (int u : adj[v]) {
        if (u != p)
            dfs(u, v);
    }

    tout[v] = ++timer;
}

bool is_ancestor(int u, int v)
{
    return tin[u] <= tin[v] && tout[u] >= tout[v];
}

int lca(int u, int v)
{
    if (is_ancestor(u, v))
        return u;
    if (is_ancestor(v, u))
        return v;
    for (int i = l; i >= 0; --i) {
        if (!is_ancestor(up[u][i], v))
            u = up[u][i];
    }
    return up[u][0];
}

void preprocess(int root) {
    tin.resize(n);
    tout.resize(n);
    timer = 0;
    l = ceil(log2(n));
    up.assign(n, vector<int>(l + 1));
    dfs(root, root);
}

int main()
{
    int nodes,ver;

    cin>>nodes>>ver;
    n=nodes;
    adj.resize(nodes);
    int e1,e2;
    for(int i=0;i<ver;i++){
        cin>>e1>>e2;
        adj[e1].pb(e2);
        adj[e2].pb(e1);
    }
    preprocess(0);
    int query;
    cin>>query;
    for(int i=0;i<query;i++){
        cin>>e1>>e2;
        cout<<lca(e1,e2);br;
    }



}


/////////////////////////////////////////////////////////////////////////////////
///
//////////                      least common ansector  in trees

bool findPath(Node *root, vector<Node *> &path, int k) 
{ 
    if (root == NULL) return false; 
    path.push_back(root); 
    if (root->data == k) 
        return true; 
    if ( (root->left && findPath(root->left, path, k)) || 
         (root->right && findPath(root->right, path, k)) ) 
        return true; 
 
    path.pop_back(); 
    return false; 
} 


Node* lca(Node* root ,int n1 ,int n2 )
{
    vector<Node *> path1, path2; 
 
    if ( !findPath(root, path1, n1) || !findPath(root, path2, n2)) 
          return newNode(-1); 

    int i; 
    for (i = 0; i < path1.size() && i < path2.size() ; i++) 
        if (path1[i]->data != path2[i]->data) 
            break; 
    return path1[i-1]; 
}

/////////////////////////////////////////////////////////////////////////////
///
//////////                  MIN Distance to reach end


        ll n;
        cin>>n;

        vector<ll> arr(n),dp(n,INT_MAX),papa_pos(n);

        for(ll i=0;i<n;i++){
            cin>>arr[i];
            papa_pos[i]=i;
        }

        dp[0]=0;

        for(ll i=0;i<n;i++){
            for(ll j=1;j<=arr[i];j++){
                if(i+j<n){
                    if(dp[i+j]> dp[i]+1){
                        dp[i+j]=dp[i]+1;
                        papa_pos[i+j]=i;
                    }
                }else{
                    break;
                }
            }
        }

        cout<<(dp[n-1]==INT_MAX ? -1 : dp[n-1]);br;



/////////////////////////////////////////////////////////////////////////////
///
//////////                  Valid Mountain Array

// fisrt inc and decreasing 

class Solution {
public:
    bool validMountainArray(vector<int>& A) {
        int i=0,j=0;
        int n=A.size();
        
        int pos=1;
        while(pos<n and A[pos-1]<A[pos] ){
            i+=1;
            pos+=1;
        }
        
        while(pos<n and A[pos-1]>A[pos]){
            j+=1;
            pos+=1;
        }
        
        cout<<i+j;
        if(i>0 and j>0 and i+j+1==n){
            return true;
        }else{
            return false;
        }
        
    }
};


/////////////////////////////////////////////////////////////////////////////
///
// Given an array A of non-negative integers, return an array consisting of all 
// the even elements of A, followed by all the odd elements of A.


vector<int> sortArrayByParity(vector<int>& A) {
        int last=0;
        int n=A.size();
        
        for(int i=0;i<n;i++){
            if(A[i]%2==0){
                swap(A[last++],A[i]);
            }
        }
        
        return A;
        
    }

/////////////////////////////////////////////////////////////////////////////
///
//////////                  Third Maximum number

//////////////////////   beats 20 %  with fast_input
class Solution {
public:
    int thirdMax(vector<int>& nums) {
        map<long,int,greater <long>> dp;
        
        for(int i:nums){
            dp[i]+=1;
        }
        
        auto ptr=dp.begin(),ptr2=dp.begin();
        
        ptr++;
        ptr++;
        
        if(ptr==dp.end()){
            return ptr2->first;
        }else{
            return ptr->first;
        }

    }
};


/////////////////////////   beats 99% with fast_input

int thirdMax(vector<int>& nums) {
    
        int max1=INT_MIN;
        int max2=INT_MIN;
        int max3=INT_MIN;
        
        bool flag1=false;
        bool flag2=false;
        bool flag3=false;
        
        int n=nums.size();
        
        for(int i=0;i<n;i++){
            if(max1<=nums[i]){
                flag1=true;
                max1=nums[i];
            }
        }
        
        for(int i=0;i<n;i++){
            if(max2<=nums[i] and nums[i]!=max1){
                flag2=true;
                max2=nums[i];
            }
        }
        
        for(int i=0;i<n;i++){
            if(max3<=nums[i] and nums[i]!=max1 and nums[i]!=max2){
                flag3=true;
                max3=nums[i];
            }
        }
        
        if(!flag3) return max1;
        
        return max3;
        
    }

/////////////////////////   Find All Numbers Disappeared in an Array

class Solution {
public:
    vector<int> findDisappearedNumbers(vector<int>& nums) {
        vector<int> ans;
        int n=nums.size();
        
        for(int x=0;x<n;x++){
            if(nums[abs(nums[x])-1]>0)
            nums[abs(nums[x])-1]=-nums[abs(nums[x])-1];
        }
        
        for(int x=0;x<n;x++){
            if(nums[x]>0){
                ans.push_back(x+1);
            }
        }
        
        
        return ans;
        
    }
};



/////////////////////////   connected island

int row,col,max_val;


vector<int> parent,ranking;

int get_num(pair<int,int> pos){
    return pos.first*col+pos.second;
}

void make_set(int i){
    parent[i]=i;
    ranking[i]=0;
}


int find_set(int i){
    if(i==parent[i]){
        return i;
    }
    else{
        return parent[i]=find_set(parent[i]);
    }
}
void join_set(int i,int j){
    int a=find_set(parent[i]);
    int b=find_set(parent[j]);
    
    
    if(a!=b){
        if (ranking[a] < ranking[b])
            swap(a, b);
        parent[b] = a;
        if (ranking[a] == ranking[b])
            ranking[a]++;
    }
    
    
}
int numIslands(vector<vector<char>>& grid) {
    row=grid.size();
    if(row==0){
        return 0;
    }
    
    col=grid[0].size();
    
    max_val=get_num({row-1,col-1})+1;
    
    parent.resize(max_val);
    ranking.resize(max_val);
    
    for(int i=0;i<max_val;i++){
        make_set(i);
    }
    
    
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            
            if(grid[i][j]=='0'){
                parent[get_num({i,j})]=-1;
            }else{
                if(i<row-1 and grid[i+1][j]=='1'){
                    join_set(get_num({i,j}),get_num({i+1,j}));
                }
                
                if(j<col-1 and grid[i][j+1]=='1'){
                    join_set(get_num({i,j}),get_num({i,j+1}));
                }
            }
        }
    }
    
    set<int> s;
    
    for(int i=0;i<max_val;i++){
        if(parent[i]==-1){
            continue;
        }
        
      //  cout<<i<<" "<<find_set(i)<<endl;
        
        s.insert(find_set(i));
        
    }
    return s.size();

    
}
};

/////////////////////////////////////////////////////////////////////////////
///
//////////                   Palindrome partition (min partition req)


#include <limits.h> 
#include <stdio.h> 
#include <string.h> 
int min(int a, int b) { return (a < b) ? a : b; } 
int minPalPartion(char* str) 
{ 
    int n = strlen(str); 
    int C[n]; 
    bool P[n][n]; 

    int i, j, k, L;  
    for (i = 0; i < n; i++) { 
        P[i][i] = true; 
    } 
    for (L = 2; L <= n; L++) {
        for (i = 0; i < n - L + 1; i++) { 
            j = i + L - 1; 
            if (L == 2) 
                P[i][j] = (str[i] == str[j]); 
            else
                P[i][j] = (str[i] == str[j]) && P[i + 1][j - 1]; 
        } 
    } 
    
    for (i = 0; i < n; i++) { 
        if (P[0][i] == true) 
            C[i] = 0; 
        else { 
            C[i] = INT_MAX; 
            for (j = 0; j < i; j++) { 
                if (P[j + 1][i] == true && 1 + C[j] < C[i]) 
                    C[i] = 1 + C[j]; 
            } 
        } 
    } 
    return C[n - 1]; 
} 
int main() 
{ 
    char str[] = "ababbbabbababa"; 
    printf("Min cuts needed for Palindrome Partitioning is %d", 
        minPalPartion(str)); 
    return 0; 
} 




