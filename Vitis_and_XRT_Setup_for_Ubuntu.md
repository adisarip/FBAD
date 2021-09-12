

# Xilinx Vitis & XRT Setup on Ubuntu

#### **WARNING: Xilinx XRT 2021.1 will NOT install in Ubuntu 20.04.3 LTS (with kernel version 5.11).**  

### Setting up the Vitis Environment with XRT toolset on Ubuntu 18.04.5:  
----
> __STEP-1: (Setting up the Ubuntu on a VM for Vitis)__  
> - Perform these steps if you are planning to install Vitis on a Virtual Machine.  
> - Download the Ubuntu __18.04.5 LTS__ installer __`.iso`__ file.  
> - Install VMWare Fusion Player (by using a personal use license which you get for free if you sign-up).  
> - Create a virtual machine with 16GB RAM & 300GB Hard disk space.  
> - Install Ubuntu 18.04.5 LTS (kernel version: __`5.4.0-81-generic`__)  
> - After installation Ubuntu software upgrade would try to upgrade the kernel to 5.11 - DENY the Upgrade.  
> - Install 'vim' & 'git'.  
> ```  
> sudo apt-get install vim  
> sudo apt-get install git  
> ```  
&nbsp;

> __STEP-1: (Setting up the OS on bare-metal for Vitis)__  
> - Perform these steps if you are planning to install Vitis on bare-metal.  
> - Install Ubuntu __18.04.5 LTS__ (kernel version: __`5.4.0-81-generic`__)  
> - After installation Ubuntu software upgrade would try to upgrade the kernel to 5.11 - DENY the Upgrade.  
> - Make sure you have minumum 16GB of RAM and 300GB of empty hard-disk space.  
> - Install 'vim' and 'git'.  
> ```  
> sudo apt-get install vim  
> sudo apt-get install git  
> ```   

----
__NOTE:__  
- If you cannot allocate or don't have 16GB RAM space then create a swapfile for the remaining size.  
- (Eg: Allocating 8GB RAM + 8GB/or >8GB of swapfile after the OS installation).  
- After creating the swapfile you can check your RAM allocation with the "free" command as below.  
```
$ free -m
              total        used        free      shared  buff/cache   available
Mem:       16369584     1420200    10733888       17864     4215496    14593528
Swap:      16777212           0    16777212
$
```
----

> __STEP-2: (Installing Vitis SDK)__  
> - Download the Vitis Unified Installer for your OS and follow the instructions to install it.  
> - Its better to download as an image and install later, because if you have the installer offline installation can be performed multiple times in case there any failures.)  
> - During installation, you can see around 200GB of your Hard-disk space will be used but finally once the installation is completed the final hard-disk space the installed Vitis software takes is only around 80GB. However, if you have created a VM, then you will see the size of your VM file has increased to >200GB (provided you haven't pre-allocated the whole 300GB of hard-disk space).  
> - Just follow the instructions from the installer and installation can be completed very easily.

&nbsp;

> __STEP-3: (Install & Setup Xilinx XRT)__  
> - Xilinx XRT is readily available with CentOS/RHEL flavours of Linux, however with Ubuntu we have to build it from the source.  
> - Download Xilinx XRT source code from github.  
> ```
> $ git clone https://github.com/Xilinx/XRT.git
> ```
> - After downloading the source __`git branch`__ command would show that you have the  __`master`__ instance of XRT downloaded. You need to change your development branch to __`2021.1`__.
> ```
> $ cd XRT
> $ get fetch origin
> $ git branch -a       <--- Check what remote branches are available
> $ git checkout origin/2021.1
> $ git pull
> ```
> - Now typing __`git branch`__ should show you are on __`2021.1`__ branch.  
> &nbsp;
> - Now run `xrtdeps.sh` script to install all the dependency softwares needed for XRT on your system.  
> ```
> $ pwd     (check you are in ~/XRT directory)
> $ sudo ./src/runtime_src/tools/scripts/xrtdeps.sh
> ```
> &nbsp;
> __Fixing the XRT Source for Linux Kernel version 5.4 (IMPORTANT)__
> ```
> $ pwd     (check you are in ~/XRT directory)
> $ vim ./src/runtime_src/core/pcie/driver/aws/kernel/mgmt/mgmt-bit.c
> Delete the line#765:    err = !access_ok(VERIFY_READ, buffer, bin_obj.m_header.m_length);  
> Add this line#765:      err = !AWSMGMT_ACCESS_OK(VERIFY_READ, buffer, bin_obj.m_header.m_length);
> ```
> - Check if you have the `dkms` installed -- __`which dkms`__.
> - This package should be installed by now. If for some reason if it is not installed then install it.  
> ```
> $ sudo apt-get install dkms
> ```
> - Export the Vitis environment variable.  
> ```
> $ export XILINX_VITIS=<Vitis_installation_directory>
> ```
> - Now, build the Xilinx XRT from source using following steps  
> ```
> $ cd ~/XRT/build
> $ ./build.sh
> ```
> - This script will build the Xilinx XRT from source for your Linux Kernel for `Debug` & `Release` targets.  
> - After the build is completed, you will have 2 directories `Debug` & `Release` inside the `build` folder. Go to Release folder -- `cd ./Release`
> - You will see lot of __`*.deb`__ files created. However, you only need to install 2 packages.
> ```
> $ sudo dpkg -i ./xrt_202110.2.11.0_18.04-amd64-xrt.deb
> $ sudo dpkg -i ./xrt_202110.2.11.0_18.04-amd64-aws.deb
> ```
> - Verify that the Xilinx XRT modules are installed properly as follows:
> ```
> $ ls -ltr /opt/xilinx/xrt                               <-- should have the XRT binaries and libraries
> $ ls -ltr /var/lib/dkms                                 <-- should display 2 folders 'xrt' & 'xrt-aws'.
> $ ls -ltr /lib/modules/5.4.0-81-generic/updates/dkms    <-- should list 3 files 'awsmgmt.ko', 'xocl.ko' & 'xclmgmt.ko'.
> ```
&nbsp;

> __STEP-4: (Get the AWS FPGA source)__  
> - You have setup Xilinx Vitis & XRT environments, however you still need the AWS FPGA platform files.
> - For that follow the following steps:  
>```
> $ cd ~/
> $ echo $XILINX_VITIS         <-- Should display the Vitis installation directory. If not set it.
> $ echo $XILINX_XRT           <-- Should be set to __`/op/xilinx/xrt`__. If not set it.
> $
> $ git clone --recursive https://github.com/aws/aws-fpga.git
> $ cd aws-fpga
> $ . ./vitis_setup.sh      <-- This will download and set the correct AWS Platform for FPGA.
>
> After the above steps you will see the following file downloaded in aws-fpga directory  
> $ ls ./Vitis/aws_platform/xilinx_aws-vu9p-f1_shell-v04261818_201920_2/hw/xilinx_aws-vu9p-f1_shell-v04261818_201920_2.xsa
>```
&nbsp;

> __STEP-5: (Setup the Vitis environment file for bash shell)__  
> - copy the following lines in a file (eg: ~/set_vitis_env).  
> - For that follow the following steps:  
> ```
> $ cat ~/set_vitis_env
> # Replace the paths accordingly based on your installation directories
> export XILINX_VITIS=/tools/Xilinx/Vitis/2021.1
> export XILINX_XRT=/opt/xilinx/xrt
> export AWS_FPGA_REPO_DIR=/home/aditya/aws-fpga
>
> # By exporting this variable you will automatically see the platform loaded in the Vitis GUI.
> export PLATFORM_REPO_PATHS=$AWS_FPGA_REPO_DIR/Vitis/aws_platform/xilinx_aws-vu9p-f1_shell-v04261818_201920_2
>
> # Sourcing the Vitis environment
> . $XILINX_VITIS/settings64.sh
> . $XILINX_XRT/setup.sh
> . $AWS_FPGA_REPO_DIR/vitis_setup.sh
> . $AWS_FPGA_REPO_DIR/vitis_runtime_setup.sh
>
> # setting the LD_LIBRARY_PATH
> export LD_LIBRARY_PATH=$XILINX_XRT/lib:$XILINX_VITIS/lib/lnx64.o  
>
> $
> $ chmod 755 ~/set_vitis_env
> $
> ```
&nbsp;
After the above steps exit from brash and re-start the system.

> __STEP-6: (Vitis Acceleration Dev Environment Init)__  
> - Initialize the environment every time when you want to work on Vitis as follows:  
> - Open a `terminal` window  and source the vitis enviroment as `. ~/set_vitis_env`. Enter sudo password if asked for.
> Vitis environment with XRT setup for FPGA Acceleration Development flow is complete.
&nbsp;

- __After the above steps are completed you are now set to execute the Vitis tutorials 1 & 2, availale at  - https://xilinx.github.io/xup_compute_acceleration  as required.__  
- __Running Emulation SW/HW will be done very fast. However, creating Hardware will take around 2/2.5 Hrs.__  
