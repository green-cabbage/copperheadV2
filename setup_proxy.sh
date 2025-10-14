# conda activate /depot/cms/kernels/root632
conda activate /depot/cms/kernels/coffea_latest
voms-proxy-init -voms cms -rfc -valid 192:00
export RUCIO_ACCOUNT=hyeonseo
export VOMS_PATH=$(echo $(voms-proxy-info | grep path) | sed 's/path.*: //')
export VOMS_USERID=$(echo $(voms-proxy-info | grep path) | sed 's/.*p_u//')
export VOMS_TRG=/depot/cms/users/$USER/x509up_u$VOMS_USERID
cp $VOMS_PATH $VOMS_TRG
echo "Your proxy is copied here: "$VOMS_TRG
export X509_USER_PROXY=$VOMS_TRG
export WORKDIR=$PWD
export XRD_REQUESTTIMEOUT=2400
source /cvmfs/cms.cern.ch/cmsset_default.sh
# for gfal commands:
voms-proxy-init -verify --rfc --voms cms -valid 192:00
source /cvmfs/oasis.opensciencegrid.org/osg-software/osg-wn-client/current/el8-x86_64/setup.sh
