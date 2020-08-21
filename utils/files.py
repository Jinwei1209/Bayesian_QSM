import os
import scipy.io as sio
import h5py
import nibabel as nib
import pydicom
import numpy as np
import datetime


def recursiveFilesWithSuffix(rootDir = '.', suffix = ''):

    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootDir)
        for filename in filenames if filename.endswith(suffix)]


def listFolders(rootDir = '.'):

    return [os.path.join(rootDir, filename)
        for filename in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir, filename))]


def listFilesWithSuffix(rootDir = '.', suffix = None):

    if suffix:
        res = [os.path.join(rootDir, filename) 
            for filename in os.listdir(rootDir) 
                if os.path.isfile(os.path.join(rootDir, filename)) and filename.endswith(suffix)]    
    else:
        res = [os.path.join(rootDir, filename) 
            for filename in os.listdir(rootDir) 
                if os.path.isfile(os.path.join(rootDir, filename))]    
    return res


# Data I/O
def load_nii(filename):
    return nib.load(filename).get_data()


def save_nii(data, filename, filename_sample=''):

    if filename_sample:
        nib.save(nib.Nifti1Image(data, None, nib.load(filename_sample).header), filename)
    else:
        nib.save(nib.Nifti1Image(data, None, None), filename)


def load_h5(filename, varname='data'):

    with h5py.File(filename, 'r') as f:
        data = f[varname][:]
    return data


def save_h5(data, filename, varname='data'):

    with h5py.File(filename, 'w') as f:
        f.create_dataset(varname, data=data)
        
        
def load_mat(filename, varname='data'):

    try:
        import scipy.io as sio
        f = sio.loadmat(filename)
        data = f[varname]        
    except:
        data = load_h5(filename, varname=varname)
        if data.ndim == 4:
            data = data.transpose(3,2,1,0)
        elif data.ndim == 3:
            data = data.transpose(2,1,0)
    return data
        
    
def load_dicom(foldername, flag_info=True):

    foldername, _, filenames = next(os.walk(foldername))
    filenames = sorted(filenames)
    data, info = [], {}
    slice_min, loc_min, slice_max, loc_max = None, None, None, None
    for filename in filenames:
        dataset = pydicom.dcmread('{0}/{1}'.format(foldername, filename))
        data.append(dataset.pixel_array)
        # Voxel size
        info['voxel_size'] = tuple(map(float, list(dataset.PixelSpacing) + [dataset.SpacingBetweenSlices]))
        # Slice location
        if slice_min is None or slice_min > float(dataset.SliceLocation):
            slice_min = float(dataset.SliceLocation)
            loc_min = np.array(dataset.ImagePositionPatient)
        if slice_max is None or slice_max < float(dataset.SliceLocation):
            slice_max = float(dataset.SliceLocation)
            loc_max = np.array(dataset.ImagePositionPatient)
    data = np.stack(data, axis=-1)
    # Matrix size
    info['matrix_size'] = data.shape
    # B0 direction
    affine2D = np.array(dataset.ImageOrientationPatient).reshape(2,3).T
    affine3D = (loc_max - loc_min) / ((info['matrix_size'][2]-1)*info['voxel_size'][2])
    affine3D = np.concatenate((affine2D, affine3D.reshape(3,1)), axis=1)
    info['B0_dir'] = tuple(np.dot(np.linalg.inv(affine3D), np.array([0, 0, 1])))
    if flag_info:
        return data, info
    else:
        return data
    
    
def save_dicom(data, foldername_tgt, foldername_src):

    if not os.path.exists(foldername_tgt):
        os.mkdir(foldername_tgt)
    foldername_src, _, filenames = next(os.walk(foldername_src))
    filenames = sorted(filenames)
    for i, filename in enumerate(filenames):
        dataset = pydicom.dcmread('{0}/{1}'.format(foldername_src, filename))
        dataset.PixelData = data[..., i].tobytes()
        dataset.save_as('{0}/{1}'.format(foldername_tgt, filename))


class Logger():

    def __init__(
        self, 
        folderName,
        rootName, 
        rsa,
        validation,
        test, 
        flagFrint=True, 
        flagSave=True,

        ):
        
        self.flagFrint = flagFrint
        self.flagSave = flagSave

        self.folderName = folderName
        self.rootName = rootName

        if(not os.path.exists(self.rootName)):
            os.mkdir(self.rootName)
        self.logPath = os.path.join(self.rootName, self.folderName)
        print(self.logPath)

        self.t0 = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
        print(self.t0)
        self.fileName = 'linear_factor={0}_validation={1}_test={2}_resnet'.format(rsa, validation, test) + '.log'
        self.filePath = os.path.join(self.logPath, self.fileName)

        if self.flagSave:
            if not os.path.exists(self.logPath):
                os.mkdir(self.logPath)
            self.file = open(self.filePath, 'w')
            self.file.write('Logs start:')
            self.file.write('\n')

    def print_and_save(self, string, *args):

        if not isinstance(string, str):
            string = str(string)

        if self.flagFrint:
            print(string % (args))

        if self.flagSave:
            self.file = open(self.filePath, 'a+')
            self.file.write(string % (args))
            self.file.write('\n')

    def close(self):

        # self.flagFrint = False
        # self.flagSave = False
        self.file.close()