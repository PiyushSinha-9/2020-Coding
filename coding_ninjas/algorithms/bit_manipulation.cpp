
int whichbit(ll n,int pos){
    if(n>>pos & 1){
        return 1;
    }else{
        return 0;
    }
}


ll turnon(ll n,ll pos){


    // ll val=1;
    // val=val<<pos;
    // n=n | val;
    // return n;

    return ((1<<pos) | n);

}


ll turnoff(ll n,ll pos){
    return ( ~(1<<pos) & n);
}


void showbits(ll n,ll size){
    for(ll i=size;i>=0;i--){
        cout<<whichbit(n,i);
    }
}


ll clearAllBits(ll n, ll i){
    
    for(int x=i;x<32;x++){
        if(n>>x & 1){
            n=turnoff(n,x);
        }
    }
    
    return n;
    
    
}



