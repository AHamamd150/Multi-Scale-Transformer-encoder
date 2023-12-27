from input import *
from func import *
from input import *
#################
def prepare_inputs(sig_dir,bkg_dir,out_dir):
    cont_pt,cont_eta,cont_phi,cont_E,kin_sig = prepare_files (sig_dir,n_cont)
    contb_pt,contb_eta,contb_phi,contb_E,kin_bkg = prepare_files (bkg_dir,n_cont)
    cont_pt = np.array(flatten(cont_pt))
    cont_eta = np.array(flatten(cont_eta))
    cont_phi = np.array(flatten(cont_phi))
    contb_pt = np.array(flatten(contb_pt))
    contb_eta = np.array(flatten(contb_eta))
    contb_phi = np.array(flatten(contb_phi))
    cont_phi = rot_phi(cont_phi)
    contb_phi = rot_phi(contb_phi)
    ##### Center the jet content ###########
    eta_c = np.copy(cont_eta)   
    phi_c = np.copy(cont_phi)
    pt_c = np.copy(cont_pt)
    etab_c = np.copy(contb_eta)   
    phib_c = np.copy(contb_phi)
    ptb_c = np.copy(contb_pt)
    for i in tqdm(range(np.array(cont_pt).shape[0])):
              eta_c[i,:],phi_c[i,:] = centre_jet(np.array(cont_eta[i]),np.array(cont_phi[i]),np.array(cont_pt[i]))
    for i in tqdm(range(np.array(contb_pt).shape[0])):
              etab_c[i,:],phib_c[i,:] = centre_jet(np.array(contb_eta[i]),np.array(cont_phi[i]),np.array(contb_pt[i]))

    ##### Rotate the jet content ########
    for i in tqdm(range(np.array(cont_pt).shape[0])):
              eta_c[i,:],phi_c[i,:] = rotate_jet(np.array(eta_c[i]),np.array(phi_c[i]),np.array(pt_c[i]))
    for i in tqdm(range(np.array(contb_pt).shape[0])):
              etab_c[i,:],phib_c[i,:] = rotate_jet(np.array(etab_c[i]),np.array(phib_c[i]),np.array(ptb_c[i]))
    ########### flip where highest momenta are in upper right corner ##########          
    for i in tqdm(range(np.array(cont_pt).shape[0])):
              eta_c[i,:],phi_c[i,:] = flip_jet(np.array(eta_c[i]),np.array(phi_c[i]),np.array(pt_c[i]))
    for i in tqdm(range(np.array(contb_pt).shape[0])):
              etab_c[i,:],phib_c[i,:] = flip_jet(np.array(etab_c[i]),np.array(phib_c[i]),np.array(ptb_c[i]))
              
    ######################
    pt_J = kin_sig[:,1]
    eta_J = kin_sig[:,2]
    phi_J = kin_sig[:,3]
    ptb_J = kin_bkg[:,1]
    etab_J = kin_bkg[:,2]
    phib_J = kin_bkg[:,3]      
    ##### Extra features to be included in the ML ######
    log_pt = np.array([np.log(pt_c[i]/pt_J[i]) for i in range(len(pt_J))])
    del_eta= np.array([(eta_c[i]-eta_J[i]) for i in range(len(eta_J))])
    del_phi= np.array([(phi_c[i]-phi_J[i]) for i in range(len(phi_J))])
    del_R = np.sqrt(del_eta**2+del_phi**2)

    logb_pt = np.array([np.log(ptb_c[i]/ptb_J[i]) for i in range(len(ptb_J))])
    delb_eta= np.array([(etab_c[i]-etab_J[i]) for i in range(len(etab_J))])
    delb_phi= np.array([(phib_c[i]-phib_J[i]) for i in range(len(phib_J))])
    delb_R = np.sqrt(delb_eta**2+delb_phi**2)    
    #######################################
    df_s = np.concatenate((np.expand_dims(eta_c,-1),np.expand_dims(phi_c,-1),np.expand_dims(pt_c,-1),np.expand_dims(log_pt,-1),np.expand_dims(del_eta,-1),np.expand_dims(del_phi,-1),np.expand_dims(del_R,-1)),axis=-1)
    df_b = np.concatenate((np.expand_dims(etab_c,-1),np.expand_dims(phib_c,-1),np.expand_dims(ptb_c,-1),np.expand_dims(logb_pt,-1),np.expand_dims(delb_eta,-1),np.expand_dims(delb_phi,-1),np.expand_dims(delb_R,-1)),axis=-1)
    df_s = mask(df_s)
    df_b = mask(df_b)
    np.savez_compressed(outdir+'/signal', df_s);
    np.savez_compressed(outdir+'/background', df_b);
    return 0




