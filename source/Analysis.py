import sys, os
import numpy as np
from input import * 
import ROOT
os.chdir('%s/'%(delphes_dir))
ROOT.gSystem.Load("libDelphes.so")

try:
    ROOT.gInterpreter.Declare('#include "%s/classes/DelphesClasses.h"'%(delphes_dir))
    ROOT.gInterpreter.Declare('#include "%s/external/ExRootAnalysis/ExRootTreeReader.h"'%(delphes_dir))
except:
    print('Delphes is not correctly linked!!.... Exit')
    sys.exit()
import glob
from tqdm import tqdm
d = dict(file=sys.stdout,colour='green')
#################################
class Analysis:
 
    ''' The analysis class to prcess root files 
        after delphes. 
        Input arg: file.root, number of events to be processed
        Output: arrays of dimension 1 for different obervables
                       
        Please note: output is cosidered after applying all the initial cuts and the files 
        are passed to ML model'''
    
    
    def __init__(self,inp, NN,n_const):
        self.inp = inp
        self.NN = NN
        self.n_const = n_const
   
    
    def phi_pi(slef,x):
        if (x > np.pi):  
            x= x -2*np.pi
        elif (x < -np.pi):
            x = x + 2*np.pi
        return x    

    def DeltaR(slef,x1,x2,y1,y2):
        return np.sqrt((x1-y1)**2+(x2-y2)**2)
    
    ''' Function to do the analysis in delphes and apply the initial cuts and event selection'''
    def Analysis_reco(self):
        chain= ROOT.TChain("Delphes")
        chain.Add(self.inp)
        # Create object of class ExRootTreeReader
        treeReader = ROOT.ExRootTreeReader(chain)
        Entries = treeReader.GetEntries()
        branchephoton = treeReader.UseBranch("Photon")
        branchJet = treeReader.UseBranch("Jet")
        branchParticle = treeReader.UseBranch("Particle")
        branchGenJet = treeReader.UseBranch("GenJet")
        event=0.0;cut_1= 0.0;cut_2 =0.0;cut_3=0.0
        Jet_m,Jet_pt,Jet_eta,Jet_phi,Jet_E = [],[],[],[],[]
        a_pt,a_eta,a_phi = [],[],[]
        hj_m,hj_pt,hj_eta,hj_phi = [],[],[],[]
        cont_pt,cont_eta,cont_phi,cont_E=[],[],[],[]
        'Loop over the considered number of events'
        for entry in tqdm(range(0,self.NN),**d,desc ='Runing delphes analysis',ascii=True):
            treeReader.ReadEntry(entry)
            event += 1
            if (branchJet.GetEntries() <1):continue
            if (branchephoton.GetEntries() <1):continue
            a1 = branchephoton.At(0)
            jet0 = []
            for jet in branchJet:
                ###############First cut#########################
                if jet.PT < 20 : continue
                if a1.PT < 20 : continue
                if abs(jet.Eta) > 2.5 : continue
                if abs(a1.Eta) > 2.5 : continue
                if (jet.P4()+a1.P4()).M() > 140: continue
                if (jet.P4()+a1.P4()).M() < 110: continue 
                cut_1 += 1     
                ###############Second cut#########################    
                if  (jet.P4().Pt() < 0.35*(jet.P4()+a1.P4()).M() ):continue
                if  (a1.P4().Pt() < 0.35*(jet.P4()+a1.P4()).M() ):continue 
                cut_2 += 1    
                ###############Third cut#########################   
                if abs(jet.Eta-a1.Eta )> 2.5 : continue
                cut_3 += 1
                  
                jet0.append(jet)                
            #######Look inside the jet###################    
            cont1_pt,cont1_eta,cont1_phi,cont1_E=[],[],[],[]
            if len(jet0) ==0.0: continue
            for ll in jet0[0].Particles:
                if (abs(ll.Eta) > 2.5):continue
                cont1_pt.append(ll.PT)
                cont1_eta.append(ll.Eta)
                cont1_phi.append(ll.Phi)
                cont1_E.append(ll.E)
                
            
                ###############Fill the containers##################
                
                
                ## Constrain the number of constituents to n_const ##
            if np.array(cont1_pt).sum() == 0.0 : continue
            Jet_m.append(jet0[0].P4().M())
            Jet_pt.append(jet0[0].P4().Pt())
            Jet_eta.append(jet0[0].P4().Eta())
            Jet_phi.append(jet0[0].P4().Phi())
            Jet_E.append(jet0[0].P4().E())

            a_pt.append(a1.P4().Pt())
            a_eta.append(a1.P4().Eta())
            a_phi.append(a1.P4().Phi())
                
            hj_m.append((jet0[0].P4()+a1.P4()).M())
            hj_pt.append((jet0[0].P4()+a1.P4()).Pt())
            hj_eta.append((jet0[0].P4()+a1.P4()).Eta())
            hj_phi.append((jet0[0].P4()+a1.P4()).Phi())    
            
            
            ########### padd the number of constitutents in each event ###############
            
            if (len(cont1_pt) < self.n_const) : 
                cont_pt.append(np.concatenate((np.array(cont1_pt),np.zeros(self.n_const-len(cont1_pt)))))
                cont_eta.append(np.concatenate((np.array(cont1_eta),np.zeros(self.n_const-len(cont1_pt)))))
                cont_phi.append(np.concatenate((np.array(cont1_phi),np.zeros(self.n_const-len(cont1_pt)))))
                cont_E.append(np.concatenate((np.array(cont1_E),np.zeros(self.n_const-len(cont1_pt)))))
            elif (len(cont1_pt) >=  self.n_const):
                cont_pt.append(cont1_pt[:self.n_const])
                cont_eta.append(cont1_eta[:self.n_const])
                cont_phi.append(cont1_phi[:self.n_const])
                cont_E.append(cont1_E[:self.n_const])
                
                
        print(f'Total number of events = {event}') 
        print('----------------------------------------')
        print(f'Number of events after cut_1 = {cut_1}')  
        print(f'Effeciency after cut_1 = {cut_1/event}') 
        print('----------------------------------------')
        print(f'Number of events after cut_2 = {cut_2}')  
        print(f'Effeciency after cut_2 = {cut_2/event}') 
        print('----------------------------------------')
        print(f'Number of events after cut_3 = {cut_3}')  
        print(f'Effeciency after cut_3 = {cut_3/event}') 
        print('----------------------------------------')
        return Jet_m,Jet_pt,Jet_eta,Jet_phi,Jet_E,a_pt,a_eta,a_phi,hj_m,hj_pt,hj_eta,hj_phi,cont_pt,cont_eta,cont_phi,cont_E
