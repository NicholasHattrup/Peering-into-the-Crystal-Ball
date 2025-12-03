import numpy as np
import os 
import sys 


equil_template = '''units lj
dimension 3
atom_style atomic
pair_style lj/cut 2.5
boundary p p p

lattice sc {rho} 
region universe block 0 13 0 13 0 13 units lattice
create_box 1 universe  
create_atoms 1 box 

mass 1 1 
pair_coeff 1 1 1 1 

write_data crystal.data
'''

prod_template = '''units lj
dimension 3
atom_style atomic
pair_style lj/cut 2.5
boundary p p p

read_data crystal.data

comm_modify cutoff 4.3 

compute myrdf all rdf 100 cutoff 4
variable xy equal pxy 
variable xz equal pxz 
variable yz equal pyz

timestep 0.002
thermo 1000

fix mynvt all nvt temp {T} {T} $(100.0*dt)

velocity all create {T} {random_seed} # randomize velocities
displace_atoms all random 0.2 0.2 0.2 {random_seed} # And positions from lattice for faster scattering

run 100000
unfix mynvt
fix mynve all nve 
run 100000


compute mymsd all msd
fix trackmsd all ave/time 1 1 500 c_mymsd[4] file {msd_file}
run 4000000'''


def main():
    rho = sys.argv[1]  # e.g., 0.92
    T = sys.argv[2]  # e.g., 0.75
    num_jobs = int(sys.argv[3])  # e.g., 100


    directory = f'rho_{rho}_T_{T}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    edited = equil_template.format(rho=rho)
    with open(os.path.join(directory, 'equil.in'), 'w') as f:
        f.write(edited)

    np.random.seed(42) # For reproducibility
    choices = np.arange(1, 10000)
    seeds = np.random.choice(choices, size=(num_jobs), replace=False)
    for idx, seed in enumerate(seeds):
        edited = prod_template.format(random_seed=seed, T=T, msd_file=f'msd_{idx+1}.dat')
        sub_directory = f'job_{idx+1}'
        path = os.path.join(directory, sub_directory)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, f'job_{idx+1}.in'), 'w') as f:
            f.write(edited)

if __name__ == "__main__":
    main()
