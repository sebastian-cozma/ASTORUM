import os
import hyperspy.api as hs
import dask.array as da
import exspy
import numpy as np
from typing import List, Tuple, Optional
import astorum.core as core
from astorum.config import *
from astorum.io import *
from astorum.utilities.PartitionTool import normalisePoissonianNoise as norm_Poisson

class SmoothNMF:
    def __init__(
        self,
        dataset : exspy.signals.EDSTEMSpectrum,
        W : Optional[np.ndarray] = None,
        H : Optional[np.ndarray] = None,
        fixedW : Optional[np.ndarray] = None,
        fixedH : Optional[np.ndarray] = None,
        gammaStepArray : Optional[np.ndarray] = None,
        components : int = 2,
        init : Optional[str] = "NNDSVD",
        maxIter : Optional[int] = 100,
        randomSeed : Optional[int] = 0,
        algorithm : Optional[str] = "LOG_SURROGATE",
        tol : Optional[np.float64] = 1E-6,
        logShift : Optional[np.float64] = 1E-14,
        eps : Optional[np.float64] = 1E-6,
        lambdaL : Optional[np.float64] = 0.0,
        mu : Optional[np.float64] = 0.0,
        epsilonReg : Optional[np.float64] = 1.0,
        dichotomyTol : Optional[np.float64] = 1E-5,
        sigmaL : Optional[np.float64] = 8.0,
        gammaStepScalar : Optional[np.float64] = 0.0,
        simplexW : Optional[bool] = True,
        simplexH : Optional[bool] = False,
        l2 : Optional[bool] = False,
        verbose : Optional[bool] = False,
        safe : Optional[bool] = False,
        debug : Optional[bool] = False,
        normalise : Optional[bool] = False,
        noStopCriterion : Optional[bool] = False,
        lineSearch : Optional[bool] = False,
        normalisePoissonianNoise : Optional[bool] = False,
        beamEnergy : Optional[int] = None,
        problemType : Optional[str] = "BREMSSTRAHLUNG",
        absorptionModelType : Optional[str] = "INTERNAL",
        azimuthAngle : Optional[np.float64] = None,
        elevationAngle : Optional[np.float64] = None,
        tiltStage : Optional[np.float64] = None,
        thickness : Optional[np.float64] = 1E-5,
        density : Optional[np.float64] = 0.0,
        widthSlope : Optional[np.float64] = 0.01,
        widthIntercept : Optional[np.float64] = 0.065,
        energyAxisSize : Optional[np.float64] = None,
        energyAxisScale : Optional[np.float64] = None,
        energyAxisOffset : Optional[np.float64] = None,
        detectorEfficiency : str = str(DETECTOR_EFFICIENCY),
        xrayDB : str = str(XRAY_200KeV),
        massAbsorptionCoefficientsFilePath : str = str(MASS_ABSORPTION_COEFFICIENTS),
        decompositionResultsFilePath : Optional[str] = "",
        absorptionMatrixFilePath : Optional[str] = "",
        thicknessMapFilePath : Optional[str] = "",
        periodicTableInfoFilePath : str = str(PERIODIC_TABLE_INFO),
        elements : Optional[List[str]] = None,
        splitLinesElements : Optional[List[str]] = None,
        energyThresholds : Optional[List[np.float64]] = None,
        quantificationElements : Optional[List[str]] = None,
        absorptionElements : Optional[List[str]] = None,
        absorptionElementsConcentrations : Optional[List[np.float64]] = None
    ):
        
        X = np.ascontiguousarray(dataset.data.reshape((dataset.data.shape[0] * dataset.data.shape[1], dataset.data.shape[2])).T, dtype = np.float64)
        
        if normalisePoissonianNoise:
            X = np.ascontiguousarray(norm_Poisson(X.T).T, dtype = np.float64)
            
        pixels = int(X.shape[1])
        
        if (beamEnergy is None):
            beamEnergy = int(dataset.metadata.Acquisition_instrument.TEM.beam_energy)
        
        if (azimuthAngle is None):
            azimuthAngle = np.float64(dataset.metadata.Acquisition_instrument.TEM.Detector.EDS.azimuth_angle)
        
        if (elevationAngle is None):
            elevationAngle = np.float64(dataset.metadata.Acquisition_instrument.TEM.Detector.EDS.elevation_angle)
        
        if (tiltStage is None):
            tiltStage = np.float64(dataset.metadata.Acquisition_instrument.TEM.Stage.tilt_alpha)
        
        if (energyAxisSize is None):
            energyAxisSize = np.float64(dataset.axes_manager[2].size)
            
        if (energyAxisScale is None):
            energyAxisScale = np.float64(dataset.axes_manager[2].scale)
            
        if (energyAxisOffset is None):
            energyAxisOffset = np.float64(dataset.axes_manager[2].offset)
            
        if (elements is None):
            elements = core.StringVector(dataset.metadata.Sample.elements)
        else:
            elements = core.StringVector(elements)
            
        if (splitLinesElements is None):
            splitLinesElements = core.StringVector([])
        else:
            splitLinesElements = core.StringVector(splitLinesElements)
        
        if (energyThresholds is None):
            energyThresholds = core.DoubleVector([])
        else:
            energyThresholds = core.DoubleVector(energyThresholds)
        
        if (quantificationElements is None):
            quantificationElements = core.StringVector([])
        else:
            quantificationElements = core.StringVector(quantificationElements)
            
        if (absorptionElements is None):
            absorptionElements = core.StringVector([])
        else:
            absorptionElements = core.StringVector(absorptionElements)
            
        if (absorptionElementsConcentrations is None):
            absorptionElementsConcentrations = np.ascontiguousarray(np.zeros(shape = (2), dtype = np.float64))
        else:
            absorptionElementsConcentrations = np.ascontiguousarray(absorptionElementsConcentrations, dtype = np.float64)
        
        Gcols = core.countGcolumns(elements, splitLinesElements) + 2
        
        if (W is None):
            W = np.ascontiguousarray(np.zeros(shape = [Gcols, components], dtype = np.float64))
            
        if (H is None):
            H = np.ascontiguousarray(np.zeros(shape = [components, pixels], dtype = np.float64))
            
        if (fixedW is None):
            fixedW = np.ascontiguousarray((-1) * np.ones(shape = [Gcols, components], dtype = np.float64))
            
        if (fixedH is None):
            fixedH = np.ascontiguousarray((-1) * np.ones(shape = [components, pixels], dtype = np.float64))
            
        if (gammaStepArray is None):
            gammaStepArray = np.ascontiguousarray(np.array([0.0, 0.0], dtype = np.float64))
            
        components = int(components)
        
        match init:
            case "RANDOM":
                init = core.SmoothNMFConstants_initialisation.RANDOM
            
            case "NNDSVD":
                init = core.SmoothNMFConstants_initialisation.NNDSVD
                
            case "NNDSVDA":
                init = core.SmoothNMFConstants_initialisation.NNDSVDA
                
            case "NNDSVDAR":
                init = core.SmoothNMFConstants_initialisation.NNDSVDAR
                
            case _:
                raise ValueError("Invalid initialisation method.")
            
        maxIter = int(maxIter)
        randomSeed = int(randomSeed)
        
        match algorithm:
            case "LOG_SURROGATE":
                algorithm = core.SmoothNMFConstants_algorithm.LOG_SURROGATE
                
            case "L2_SURROGATE":
                algorithm = core.SmoothNMFConstants_algorithm.L2_SURROGATE
                
            case "PROJECTED_GRADIENT":
                algorithm = core.SmoothNMFConstants_algorithm.PROJECTED_GRADIENT
                
            case "BMD":
                algorithm = core.SmoothNMFConstants_algorithm.BMD
                
            case _:
                raise ValueError("Invalid algorithm.")
                
        tol = np.float64(tol)
        logShift = np.float64(logShift)
        eps = np.float64(eps)
        lambdaL = np.float64(lambdaL)
        mu = np.float64(mu)
        epsilonReg = np.float64(epsilonReg)
        dichotomyTol = np.float64(dichotomyTol)
        sigmaL = np.float64(sigmaL)
        gammaStepScalar = np.float64(gammaStepScalar)
        
        simplexW = bool(simplexW)
        simplexH = bool(simplexH)
        l2 = bool(l2)
        verbose = bool(verbose)
        safe = bool(safe)
        debug = bool(debug)
        normalise = bool(normalise)
        noStopCriterion = bool(noStopCriterion)
        lineSearch = bool(lineSearch)
        
        match problemType:
            case "IDENTITY":
                problemType = core.EDXSModelConstants_problemType.IDENTITY
                
            case "NO_BREMSSTRAHLUNG":
                problemType = core.EDXSModelConstants_problemType.NO_BREMSSTRAHLUNG
        
            case "BREMSSTRAHLUNG":
                problemType = core.EDXSModelConstants_problemType.BREMSSTRAHLUNG
                
            case _:
                raise ValueError("Invalid problem type.")
            
        match absorptionModelType:
            case "INTERNAL":
                absorptionModelType = core.EDXSModelConstants_absorptionModelType.INTERNAL
                
            case "EXTERNAL":
                absorptionModelType = core.EDXSModelConstants_absorptionModelType.EXTERNAL
                
            case "NONE":
                absorptionModelType = core.EDXSModelConstants_absorptionModelType.NONE
                
            case _:
                raise ValueError("Invalid absorption model type.")
            
        thickness = np.float64(thickness)
        density = np.float64(density)
        widthSlope = np.float64(widthSlope)
        widthIntercept = np.float64(widthIntercept)
        
        detectorEfficiency = str(detectorEfficiency)
        xrayDB = str(xrayDB)
        massAbsorptionCoefficientsFilePath = str(massAbsorptionCoefficientsFilePath)
        decompositionResultsFilePath = str(decompositionResultsFilePath)
        absorptionMatrixFilePath = str(absorptionMatrixFilePath)
        thicknessMapFilePath = str(thicknessMapFilePath)
        periodicTableInfoFilePath = str(periodicTableInfoFilePath)
        
        snmf = core.SmoothNMF(X, W, H, fixedW, fixedH, gammaStepArray, 
                                    components, init, maxIter, randomSeed, algorithm, 
                                    tol, logShift, eps, lambdaL, mu, epsilonReg, dichotomyTol, sigmaL, gammaStepScalar, 
                                    simplexW, simplexH, l2, verbose, safe, debug, normalise, noStopCriterion, lineSearch)
        
        snmf.initialiseModel(beamEnergy, problemType, absorptionModelType, 
                            azimuthAngle, elevationAngle, tiltStage, thickness, density, widthSlope, widthIntercept, 
                            energyAxisSize, energyAxisScale, energyAxisOffset, 
                            detectorEfficiency, xrayDB, massAbsorptionCoefficientsFilePath, decompositionResultsFilePath, 
                            absorptionMatrixFilePath, thicknessMapFilePath, periodicTableInfoFilePath, 
                            elements, splitLinesElements, energyThresholds, quantificationElements, absorptionElements, absorptionElementsConcentrations)
        
        self.estimator = snmf
        self.dataset = dataset
        self.Q = None
    
    
    def fitTransform(self) -> None:
        self.estimator.fitTransform()
        
        
    def getDecompositionResults(self, outputDirectory : Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        G = self.estimator.G
        W = self.estimator.W
        H = self.estimator.H
        
        if (outputDirectory is not None):
            writeArrayToFile(outputDirectory + "/G.onmf", G)
            writeArrayToFile(outputDirectory + "/W.onmf", W)
            writeArrayToFile(outputDirectory + "/H.onmf", H)
            
        spectra = hs.signals.Signal1D(np.matmul(G, W).T)
        loadings = hs.signals.Signal2D(H.reshape((H.shape[0], self.dataset.data.shape[0], self.dataset.data.shape[1])))

        Q = None
        
        if (len(list(self.estimator.model.quantificationElements)) == 0):
            Q = self.estimator.model.generateQuantificationMatrix(W, H)
        else:
            Q = self.estimator.model.generateQuantificationMatrix(W, H, self.estimator.model.quantificationElements)
            
        self.Q = Q
        quant = hs.signals.Signal2D(Q.reshape((Q.shape[0], self.dataset.data.shape[0], self.dataset.data.shape[1])))
        
        return (spectra, loadings, quant)
    
    
    def computeDensityMap(self, Q : np.ndarray = None, elements : List[str] = None) -> np.ndarray:
        if (Q is not None and elements is not None):
            Q = np.ascontiguousarray(Q, dtype = np.float64)
            elements = core.StringVector(elements)
            D = self.estimator.model.computeDensityMap(Q, elements)
            
        else:
            W = self.estimator.W
            H = self.estimator.H
            D = self.estimator.model.computeDensityMap(W, H)
        
        return D
 
 
    def computeMeanAtomicNumberMap(self, Q : np.ndarray = None, elements : List[str] = None) -> np.ndarray:
        if (Q is not None and elements is not None):
            Q = np.ascontiguousarray(Q, dtype = np.float64)
            elements = core.StringVector(elements)
            meanZ = self.estimator.model.computeMeanAtomicNumberMap(Q, elements)
            
        else:
            W = self.estimator.W
            H = self.estimator.H
            meanZ = self.estimator.model.computeMeanAtomicNumberMap(W, H)
            
        return meanZ
    
    
    def optimiseQuantificationData(
        self, 
        HAADF : hs.signals.Signal2D,
        gamma : np.float64 = 1.6, 
        lambdaHAADF : np.float64 = None, 
        lambdaChem : np.float64 = 0.08, 
        lambdaTV : np.float64 = 0.15, 
        epsilon : np.float64 = 0.24, 
        nIter : int = 30, 
        nIterTV : int = 3, 
        regularise : bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        if (self.Q is None):
            raise ValueError("Quantification matrix has not been computed. Run getDecompositionResults() first.")
        
        if (lambdaHAADF is None):
            lambdaHAADF = 1 / self.Q.shape[0]
        
        gamma = np.float64(gamma)
        lambdaHAADF = np.float64(lambdaHAADF)
        lambdaChem = np.float64(lambdaChem)
        lambdaTV = np.float64(lambdaTV)
        epsilon = np.float64(epsilon)
        nIter = int(nIter)
        nIterTV = int(nIterTV)
        regularise = bool(regularise)
        
        xDim = int(HAADF.data.shape[0])
        yDim = int(HAADF.data.shape[1])
        XInit = np.ascontiguousarray(HAADF.data.reshape((HAADF.data.shape[0] * HAADF.data.shape[1])), dtype = np.float64)
        self.Q = np.ascontiguousarray(self.Q, dtype = np.float64)
        costHAADF = np.ascontiguousarray(np.zeros(shape = (nIter), dtype = np.float64))
        costChem = np.ascontiguousarray(np.zeros(shape = (nIter), dtype = np.float64))
        costTV = np.ascontiguousarray(np.zeros(shape = (nIter), dtype = np.float64))
        
        fusion = core.HAADFDataset(xDim, yDim, XInit)
        fusion.loadEDXSQuantificationData(self.Q, self.estimator.model.quantificationElements, self.estimator.model.periodicTableInfoFilePath)
        fusion.runQuantificationDataOptimisationRoutine(gamma, lambdaHAADF, lambdaChem, lambdaTV, epsilon, nIter, nIterTV, costHAADF, costChem, costTV, regularise)
        
        QOptimised = fusion.Q
        XOptimised = fusion.X
        
        return (QOptimised, XOptimised, costHAADF, costChem, costTV)
        
    
    def computeThicknessMap(self, LL : exspy.signals.EELSSpectrum, ZLP_threshold : np.float64, outputFilePath : str, D : np.ndarray = None, meanZ : np.ndarray = None, useOpenMP : Optional[bool] = True, numThreads : Optional[str] = None) -> np.ndarray:
        X = np.ascontiguousarray(LL.data.reshape((LL.data.shape[0] * LL.data.shape[1], LL.data.shape[2])).T, dtype = np.float64)
        energyAxis = np.ascontiguousarray(LL.axes_manager.signal_axes[0].axis, dtype = np.float64)
        energyAxisScale = np.float64(LL.axes_manager.signal_axes[0].scale)
        energyAxisOffset = np.float64(LL.axes_manager.signal_axes[0].offset)
        beamEnergy = np.float64(LL.metadata.Acquisition_instrument.TEM.beam_energy)
        alpha = np.float64(LL.metadata.Acquisition_instrument.TEM.convergence_angle)
        beta = np.float64(LL.metadata.Acquisition_instrument.TEM.Detector.EELS.collection_angle)
        ZLP_threshold = np.float64(ZLP_threshold)
        
        if (useOpenMP):
            if (numThreads is None):
                numThreads = str(os.cpu_count())
            
            os.environ['OMP_NUM_THREADS'] = numThreads
        
        if (D is None and meanZ is None):
            raise ValueError("Either density map or mean atomic number map must be provided.")
        
        if (D is not None and meanZ is not None):
            raise ValueError("Density map and mean atomic number map cannot be provided simultaneously.")
        
        electronMeanFreePathComputation = None
        lambdaComputationMap = None
        
        if (D is not None):
            lambdaComputationMap = np.ascontiguousarray(D, dtype = np.float64)
            electronMeanFreePathComputation = core.EELSModelConstants_electronMeanFreePathComputation.DENSITY_OF_MIXTURE
            
        elif (meanZ is not None):
            lambdaComputationMap = np.ascontiguousarray(meanZ, dtype = np.float64)
            electronMeanFreePathComputation = core.EELSModelConstants_electronMeanFreePathComputation.MEAN_ATOMIC_NUMBER
        
        EELS = core.EELSDataset(X, energyAxis, energyAxisScale, energyAxisOffset, beamEnergy, electronMeanFreePathComputation, lambdaComputationMap, alpha, beta, ZLP_threshold)
        EELS.computeThicknessMap()
        
        writeArrayToFile(outputFilePath, EELS.T)
        
        return EELS.T
        
    
    def computeAbsorptionCorrectionMatrix(self, thicknessMapFilePath : str = None, Q : np.ndarray = None, elements : List[str] = None, outputDirectory : str = None, useOpenMP : Optional[bool] = True, numThreads : Optional[str] = None) -> np.ndarray:
        if (thicknessMapFilePath is None and (Q is None or elements is None)):
            raise ValueError("Either thickness map or quantification data must be provided.")
        
        if (thicknessMapFilePath is not None and (Q is not None or elements is not None)):
            raise ValueError("Thickness map and quantification data cannot be provided simultaneously.")
        
        if (outputDirectory is None):
            outputDirectory = os.getcwd()
            
        if (useOpenMP):
            if (numThreads is None):
                numThreads = str(os.cpu_count())
            
            os.environ['OMP_NUM_THREADS'] = numThreads
        
        if (thicknessMapFilePath is not None):
            self.estimator.model.thicknessMapFilePath = thicknessMapFilePath
            A = self.estimator.model.generateAbsorptionCorrectionMatrix(self.estimator.W, self.estimator.H)
        
        elif (Q is not None and elements is not None):
            Q = np.ascontiguousarray(Q, dtype = np.float64)
            elements = core.StringVector(elements)
            A = self.estimator.model.generateAbsorptionCorrectionMatrix(Q, elements)        
        
        writeArrayToFile(outputDirectory + "/A.onmf", A)
        
        return A
    
    
    def calculateResidual(self, nSelectedComponents : Optional[int] = None) -> exspy.signals.EDSTEMSpectrum:
        X = self.estimator.X
        G = self.estimator.G
        W = self.estimator.W
        H = self.estimator.H
        
        if (nSelectedComponents is None):
            nSelectedComponents = int(W.shape[1])
        
        GW = np.matmul(G, W[:, :nSelectedComponents])
        nChannels = G.shape[0]
        
        X_R = np.matmul(GW, H[:nSelectedComponents, :])
        X_R = X_R / np.linalg.norm(X_R)
        X = X / np.linalg.norm(X)
        
        R = X - X_R
        R = R.reshape((nChannels, self.dataset.data.shape[0], self.dataset.data.shape[1])) 
        R[R <= 0.0] = 1e-14
        
        R_signal = exspy.signals.EDSTEMSpectrum(R.transpose(1, 2, 0))
            
        return R_signal 


    def printConcentrationReport(self, fitError : Optional[bool] = True):
        core.printConcentrationReport(self.estimator.G, self.estimator.W, self.estimator.H, self.estimator.model.modelElements, self.estimator.model.quantificationElements, fitError)
        
        
        
class BlockWiseSmoothNMF:
    def writeXBlocks(self, dataArray : np.ndarray, outputDirectory : str) -> None:
        blockIndex = 0

        for i in range(dataArray.blocks.shape[0]):
            for j in range(dataArray.blocks.shape[1]):
                    XFilename = outputDirectory + "/X/X_block_" + str(blockIndex) + ".inmf"
                    temp_array = dataArray.blocks[i, j, 0].compute()
                    temp_array = temp_array.reshape((temp_array.shape[0] * temp_array.shape[1], temp_array.shape[2]))
                    temp_array = temp_array.T
                    writeArrayToFile(XFilename, temp_array)
                    blockIndex += 1
                    
                    
    def createDirectories(self, inputDir : str, outputDir : str) -> None:
        pathX = os.path.join(inputDir, "X")
        pathT = os.path.join(inputDir, "T")
        os.makedirs(pathX, exist_ok = True)
        os.makedirs(pathT, exist_ok = True)
        
        pathW = os.path.join(outputDir, "W")
        pathH = os.path.join(outputDir, "H")
        pathQ = os.path.join(outputDir, "Q")
        pathA = os.path.join(outputDir, "A")
        os.makedirs(pathW, exist_ok = True)
        os.makedirs(pathH, exist_ok = True)
        os.makedirs(pathQ, exist_ok = True)
        os.makedirs(pathA, exist_ok = True)
           
                
    def __init__(
        self,
        dataset : exspy.signals.LazyEDSTEMSpectrum,
        workingDirectory : str,
        blockShape : Tuple[int, int],
        nClusters : int,
        componentsVector : np.ndarray,
        normalisePoissonianNoise : Optional[bool] = False,
        beamEnergy : Optional[int] = None,
        problemType : Optional[str] = "BREMSSTRAHLUNG",
        absorptionModelType : Optional[str] = "INTERNAL",
        azimuthAngle : Optional[np.float64] = None,
        elevationAngle : Optional[np.float64] = None,
        tiltStage : Optional[np.float64] = None,
        thickness : Optional[np.float64] = 1E-5,
        density : Optional[np.float64] = 0.0,
        widthSlope : Optional[np.float64] = 0.01,
        widthIntercept : Optional[np.float64] = 0.065,
        energyAxisSize : Optional[np.float64] = None,
        energyAxisScale : Optional[np.float64] = None,
        energyAxisOffset : Optional[np.float64] = None,
        detectorEfficiency : str = str(DETECTOR_EFFICIENCY),
        xrayDB : str = str(XRAY_200KeV),
        massAbsorptionCoefficientsFilePath : str = str(MASS_ABSORPTION_COEFFICIENTS),
        decompositionResultsFilePath : Optional[str] = "",
        absorptionMatrixFilePath : Optional[str] = "",
        thicknessMapFilePath : Optional[str] = "",
        periodicTableInfoFilePath : str = str(PERIODIC_TABLE_INFO),
        elements : Optional[List[str]] = None,
        splitLinesElements : Optional[List[str]] = None,
        energyThresholds : Optional[List[np.float64]] = None,
        quantificationElements : Optional[List[str]] = None,
        absorptionElements : Optional[List[str]] = None,
        absorptionElementsConcentrations : Optional[List[np.float64]] = None
    ) -> None:

        dataset.rechunk((blockShape[0], blockShape[1], dataset.data.shape[2]))
        X = dataset.data.rechunk((blockShape[0], blockShape[1], dataset.data.shape[2]))
        
        if normalisePoissonianNoise:
            for i in range(dataset.blocks.shape[0]):
                for j in range(dataset.blocks.shape[1]):
                    temp_array = dataset.blocks[i, j, 0].compute().reshape(temp_array.shape[0] * temp_array.shape[1], temp_array.shape[2])
                    temp_array_norm = norm_Poisson(temp_array).reshape(temp_array.shape[0], temp_array.shape[1], temp_array.shape[2])
                    X.blocks[i, j, 0] = temp_array_norm
                    
        
        inputDir = os.path.join(workingDirectory, "blockwise_input")
        outputDir = os.path.join(workingDirectory, "blockwise_output")
        os.makedirs(inputDir, exist_ok = True)
        os.makedirs(outputDir, exist_ok = True)
        
        inputDir = str(inputDir)
        outputDir = str(outputDir)
        
        self.createDirectories(inputDir, outputDir)
        self.writeXBlocks(X, inputDir)
        
        blockStructure = (int(dataset.data.shape[0] / blockShape[0]), int(dataset.data.shape[1] / blockShape[1]))
        blocks = int((dataset.data.shape[0] / blockShape[0]) * (dataset.data.shape[1] / blockShape[1]))
        blockHeight = int(blockShape[0])
        blockWidth = int(blockShape[1])
        nClusters = int(nClusters)
        
        componentsVector = np.ascontiguousarray(componentsVector, dtype = np.int32)

        if (beamEnergy is None):
            beamEnergy = int(dataset.metadata.Acquisition_instrument.TEM.beam_energy)
        
        if (azimuthAngle is None):
            azimuthAngle = np.float64(dataset.metadata.Acquisition_instrument.TEM.Detector.EDS.azimuth_angle)
        
        if (elevationAngle is None):
            elevationAngle = np.float64(dataset.metadata.Acquisition_instrument.TEM.Detector.EDS.elevation_angle)
        
        if (tiltStage is None):
            tiltStage = np.float64(dataset.metadata.Acquisition_instrument.TEM.Stage.tilt_alpha)
        
        if (energyAxisSize is None):
            energyAxisSize = np.float64(dataset.axes_manager[2].size)
            
        if (energyAxisScale is None):
            energyAxisScale = np.float64(dataset.axes_manager[2].scale)
            
        if (energyAxisOffset is None):
            energyAxisOffset = np.float64(dataset.axes_manager[2].offset)
            
        if (elements is None):
            elements = core.StringVector(dataset.metadata.Sample.elements)
        else:
            elements = core.StringVector(elements)
            
        if (splitLinesElements is None):
            splitLinesElements = core.StringVector([])
        else:
            splitLinesElements = core.StringVector(splitLinesElements)
        
        if (energyThresholds is None):
            energyThresholds = core.DoubleVector([])
        else:
            energyThresholds = core.DoubleVector(energyThresholds)
            
        if (quantificationElements is None):
            quantificationElements = core.StringVector([])
        else:
            quantificationElements = core.StringVector(quantificationElements)
            
        if (absorptionElements is None):
            absorptionElements = core.StringVector([])
        else:
            absorptionElements = core.StringVector(absorptionElements)
            
        if (absorptionElementsConcentrations is None):
            absorptionElementsConcentrations = np.ascontiguousarray(np.zeros(shape = (2), dtype = np.float64))
        else:
            absorptionElementsConcentrations = np.ascontiguousarray(absorptionElementsConcentrations, dtype = np.float64)
        
        match problemType:
            case "IDENTITY":
                problemType = core.EDXSModelConstants_problemType.IDENTITY
                
            case "NO_BREMSSTRAHLUNG":
                problemType = core.EDXSModelConstants_problemType.NO_BREMSSTRAHLUNG
        
            case "BREMSSTRAHLUNG":
                problemType = core.EDXSModelConstants_problemType.BREMSSTRAHLUNG
                
            case _:
                raise ValueError("Invalid problem type.")
            
        match absorptionModelType:
            case "INTERNAL":
                absorptionModelType = core.EDXSModelConstants_absorptionModelType.INTERNAL
                
            case "EXTERNAL":
                absorptionModelType = core.EDXSModelConstants_absorptionModelType.EXTERNAL
                
            case "NONE":
                absorptionModelType = core.EDXSModelConstants_absorptionModelType.NONE
                
            case _:
                raise ValueError("Invalid absorption model type.")
            
        thickness = np.float64(thickness)
        density = np.float64(density)
        widthSlope = np.float64(widthSlope)
        widthIntercept = np.float64(widthIntercept)
        
        detectorEfficiency = str(detectorEfficiency)
        xrayDB = str(xrayDB)
        massAbsorptionCoefficientsFilePath = str(massAbsorptionCoefficientsFilePath)
        decompositionResultsFilePath = str(decompositionResultsFilePath)
        absorptionMatrixFilePath = str(absorptionMatrixFilePath)
        thicknessMapFilePath = str(thicknessMapFilePath)
        periodicTableInfoFilePath = str(periodicTableInfoFilePath)
        
        bwsnmf = core.BlockWiseSmoothNMF(inputDir, outputDir, blockStructure[0], blockStructure[1], blockWidth, blockHeight, nClusters, componentsVector)
        
        bwsnmf.initialiseModel(beamEnergy, problemType, absorptionModelType, 
                    azimuthAngle, elevationAngle, tiltStage, thickness, density, widthSlope, widthIntercept, 
                    energyAxisSize, energyAxisScale, energyAxisOffset, 
                    detectorEfficiency, xrayDB, massAbsorptionCoefficientsFilePath, decompositionResultsFilePath, 
                    absorptionMatrixFilePath, thicknessMapFilePath, periodicTableInfoFilePath, 
                    elements, splitLinesElements, energyThresholds, quantificationElements, absorptionElements, absorptionElementsConcentrations)
        
        self.estimator = bwsnmf
        self.dataset = dataset
        self.blocks = blocks
        self.blockStructure = blockStructure
        
        
    def fitTransform(
        self,
        spectralClassificationMethod : str = "REFINED_VCA",
        spatialComputationApproach : str = "BLOCKWISE",
        spatialComputationAlgorithm : str = "SVD",
        gammaStepArray : Optional[np.ndarray] = None,
        init : Optional[str] = "NNDSVD",
        maxIter : Optional[int] = 100,
        randomSeed : Optional[int] = 0,
        algorithm : Optional[str] = "LOG_SURROGATE",
        tol : Optional[np.float64] = 1E-6,
        logShift : Optional[np.float64] = 1E-14,
        eps : Optional[np.float64] = 1E-6,
        lambdaL : Optional[np.float64] = 0.0,
        mu : Optional[np.float64] = 0.0,
        epsilonReg : Optional[np.float64] = 1.0,
        dichotomyTol : Optional[np.float64] = 1E-5,
        sigmaL : Optional[np.float64] = 8.0,
        gammaStepScalar : Optional[np.float64] = 0.0,
        simplexW : Optional[bool] = True,
        simplexH : Optional[bool] = False,
        l2 : Optional[bool] = False,
        verbose : Optional[bool] = False,
        safe : Optional[bool] = False,
        debug : Optional[bool] = False,
        normalise : Optional[bool] = False,
        noStopCriterion : Optional[bool] = False,
        lineSearch : Optional[bool] = False,
        separationOrder : Optional[np.ndarray] = None,
        precomputedW : bool = False,
        writeWBlocks : bool = True,
        clusteringSigma : Optional[np.float64] = 0.5,
        clusteringIter : Optional[int] = 100,
        clusteringTolerance : Optional[np.float64] = 1E-10,
        metric : Optional[str] = "MANHATTAN",
        p : Optional[int] = 1,
        useOpenMP : Optional[bool] = True,
        numThreads : Optional[str] = None
    ) -> None:
        
        if (gammaStepArray is None):
            gammaStepArray = np.ascontiguousarray(np.array([0.0, 0.0], dtype = np.float64))
            
        match init:
            case "RANDOM":
                init = core.SmoothNMFConstants_initialisation.RANDOM
            
            case "NNDSVD":
                init = core.SmoothNMFConstants_initialisation.NNDSVD
                
            case "NNDSVDA":
                init = core.SmoothNMFConstants_initialisation.NNDSVDA
                
            case "NNDSVDAR":
                init = core.SmoothNMFConstants_initialisation.NNDSVDAR
                
            case _:
                raise ValueError("Invalid initialisation method.")
            
        maxIter = int(maxIter)
        randomSeed = int(randomSeed)
        
        match algorithm:
            case "LOG_SURROGATE":
                algorithm = core.SmoothNMFConstants_algorithm.LOG_SURROGATE
                
            case "L2_SURROGATE":
                algorithm = core.SmoothNMFConstants_algorithm.L2_SURROGATE
                
            case "PROJECTED_GRADIENT":
                algorithm = core.SmoothNMFConstants_algorithm.PROJECTED_GRADIENT
                
            case "BMD":
                algorithm = core.SmoothNMFConstants_algorithm.BMD
                
            case _:
                raise ValueError("Invalid algorithm.")
                
        tol = np.float64(tol)
        logShift = np.float64(logShift)
        eps = np.float64(eps)
        lambdaL = np.float64(lambdaL)
        mu = np.float64(mu)
        epsilonReg = np.float64(epsilonReg)
        dichotomyTol = np.float64(dichotomyTol)
        sigmaL = np.float64(sigmaL)
        gammaStepScalar = np.float64(gammaStepScalar)
        
        simplexW = bool(simplexW)
        simplexH = bool(simplexH)
        l2 = bool(l2)
        verbose = bool(verbose)
        safe = bool(safe)
        debug = bool(debug)
        normalise = bool(normalise)
        noStopCriterion = bool(noStopCriterion)
        lineSearch = bool(lineSearch)
        
        if (separationOrder is None):
            separationOrder = np.ascontiguousarray(np.ones(shape = (self.estimator.blocks), dtype = np.int32))
            
        precomputedW = bool(precomputedW)
        writeWBlocks = bool(writeWBlocks)
        
        clusteringSigma = np.float64(clusteringSigma)
        clusteringIter = int(clusteringIter)
        clusteringTolerance = np.float64(clusteringTolerance)
        
        match metric:
            case "EUCLIDEAN":
                metric = core.BlockWiseSmoothNMFConstants_distanceMetric.EUCLIDEAN
                p = int(2)
                
            case "MANHATTAN":
                metric = core.BlockWiseSmoothNMFConstants_distanceMetric.MANHATTAN
                p = int(1)
                
            case "MAHALANOBIS":
                metric = core.BlockWiseSmoothNMFConstants_distanceMetric.MAHALANOBIS
                
            case "MINKOWSKI":
                metric = core.BlockWiseSmoothNMFConstants_distanceMetric.MINKOWSKI
                p = int(p)
                
            case "COSINE":
                metric = core.BlockWiseSmoothNMFConstants_distanceMetric.COSINE
                
            case "CHEBYSHEV":
                metric = core.BlockWiseSmoothNMFConstants_distanceMetric.CHEBYSHEV
            
            case "HAMMING":
                metric = core.BlockWiseSmoothNMFConstants_distanceMetric.HAMMING
                
            case "JACCARD":
                metric = core.BlockWiseSmoothNMFConstants_distanceMetric.JACCARD
                
            case _:
                raise ValueError("Invalid metric.")
            
        if (useOpenMP):
            if (numThreads is None):
                numThreads = str(os.cpu_count())
                
            os.environ['OMP_NUM_THREADS'] = numThreads
            
        self.estimator.computeWBlocks(gammaStepArray,
                            init, maxIter, randomSeed, algorithm, 
                            tol, logShift, eps, lambdaL, mu, epsilonReg, dichotomyTol, sigmaL, gammaStepScalar, 
                            simplexW, simplexH, l2, verbose, safe, debug, normalise, noStopCriterion, lineSearch, separationOrder,
                            precomputedW, writeWBlocks)
        
        if (spectralClassificationMethod == "SPECTRAL_CLUSTERING" or spectralClassificationMethod == "REFINED_SPECTRAL_CLUSTERING"):
            self.estimator.spectralClustering(clusteringSigma, metric, p, clusteringIter, clusteringTolerance)
            self.estimator.computeClusteredW()
            
            if (spectralClassificationMethod == "REFINED_SPECTRAL_CLUSTERING"):
                self.estimator.refineW(gammaStepArray,
                                    init, maxIter, randomSeed, algorithm, 
                                    tol, logShift, eps, lambdaL, mu, epsilonReg, dichotomyTol, sigmaL, gammaStepScalar, 
                                    simplexW, simplexH, l2, verbose, safe, debug, normalise, noStopCriterion, lineSearch)
            
        elif (spectralClassificationMethod == "VCA" or spectralClassificationMethod == "REFINED_VCA"):
            self.estimator.VCA(verbose)
            
            if (spectralClassificationMethod == "REFINED_VCA"):
                self.estimator.refineWVCA(gammaStepArray,
                                    init, maxIter, randomSeed, algorithm, 
                                    tol, logShift, eps, lambdaL, mu, epsilonReg, dichotomyTol, sigmaL, gammaStepScalar, 
                                    simplexW, simplexH, l2, verbose, safe, debug, normalise, noStopCriterion, lineSearch)
        
        
        if (spatialComputationApproach == "BLOCKWISE"):
            if (spatialComputationAlgorithm == "SVD"):
                self.estimator.computeHBlocksSVD()
                    
            elif (spatialComputationAlgorithm == "SMOOTHNMF"):
                self.estimator.computeHBlocks(gammaStepArray,
                                    init, maxIter, randomSeed, algorithm,
                                    tol, logShift, eps, lambdaL, mu, epsilonReg, dichotomyTol, sigmaL, gammaStepScalar,
                                    simplexW, simplexH, l2, verbose, safe, debug, normalise, noStopCriterion, lineSearch)
                
        elif (spatialComputationApproach == "MONOLITHIC"):
            if (spatialComputationAlgorithm == "SVD"):
                self.estimator.computeHMatrixSVD()
                    
            elif (spatialComputationAlgorithm == "SMOOTHNMF"):
                self.estimator.computeHMatrix(gammaStepArray,
                                    init, maxIter, randomSeed, algorithm,
                                    tol, logShift, eps, lambdaL, mu, epsilonReg, dichotomyTol, sigmaL, gammaStepScalar,
                                    simplexW, simplexH, l2, verbose, safe, debug, normalise, noStopCriterion, lineSearch)
                
        writeArrayToFile(self.estimator.outputDir + "/G.onmf", self.estimator.G)
        
        self.spatialComputationApproach = spatialComputationApproach
          
        
    def readHBlocks(self) -> np.ndarray:
        H = np.zeros((int(self.estimator.nClusters), int(self.blockStructure[0] * self.estimator.blockHeight), int(self.blockStructure[1] * self.estimator.blockWidth)))
        
        for i in range(int(self.blockStructure[0])):
            for j in range(int(self.blockStructure[1])):
                HFilename = self.estimator.outputDir + "/H/H_block_" + str(i * int(self.blockStructure[1]) + j) + ".onmf"
                temp_array = readArrayFromFile(HFilename)
                temp_array = temp_array.reshape((int(self.estimator.nClusters), int(self.estimator.blockHeight), int(self.estimator.blockWidth)))
                H[:, i * int(self.estimator.blockHeight) : (i + 1) * int(self.estimator.blockHeight), j * int(self.estimator.blockWidth) : (j + 1) * int(self.estimator.blockWidth)] = temp_array
        
        H = np.nan_to_num(H, nan = 1E-14)
                
        return H


    def readQBlocks(self) -> np.ndarray:
        nElements = 0
        
        if (len(list(self.estimator.model.quantificationElements)) == 0):
            nElements = len(list(self.estimator.model.elements))
        else:
            nElements = len(list(self.estimator.model.quantificationElements))
        
        Q = np.zeros((nElements, int(self.blockStructure[0] * self.estimator.blockHeight), int(self.blockStructure[1] * self.estimator.blockWidth)))
        
        for i in range(int(self.blockStructure[0])):
            for j in range(int(self.blockStructure[1])):
                QFilename = self.estimator.outputDir + "/Q/Q_block_" + str(i * int(self.blockStructure[1]) + j) + ".onmf"
                temp_array = readArrayFromFile(QFilename)
                temp_array = temp_array.reshape((nElements, int(self.estimator.blockHeight), int(self.estimator.blockWidth)))
                Q[:, i * int(self.estimator.blockHeight) : (i + 1) * int(self.estimator.blockHeight), j * int(self.estimator.blockWidth) : (j + 1) * int(self.estimator.blockWidth)] = temp_array
                
        Q = np.nan_to_num(Q, nan = 1E-14)
                
        return Q
            
    
    def getDecompositionResults(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        G = readArrayFromFile(self.estimator.outputDir + "/G.onmf")
        W = readArrayFromFile(self.estimator.outputDir + "/W/W_clustered.onmf")
        
        if (self.spatialComputationApproach == "BLOCKWISE"):
            H = self.readHBlocks()
            Q = self.readQBlocks()
        
        elif (self.spatialComputationApproach == "MONOLITHIC"):
            H = readArrayFromFile(self.estimator.outputDir + "/H/H_monolithic.onmf")
            H = H.reshape((H.shape[0], self.dataset.data.shape[0], self.dataset.data.shape[1]))
            Q = readArrayFromFile(self.estimator.outputDir + "/Q/Q_monolithic.onmf")
            Q = Q.reshape((Q.shape[0], self.dataset.data.shape[0], self.dataset.data.shape[1]))
        
        spectra = hs.signals.Signal1D(np.matmul(G, W).T)
        loadings = hs.signals.Signal2D(H)
        quant = hs.signals.Signal2D(Q)
        
        return (spectra, loadings, quant)
    
    
    def computeDensityMaps(self) -> np.ndarray:
        D = np.zeros((int(self.blockStructure[0] * self.estimator.blockHeight), int(self.blockStructure[1] * self.estimator.blockWidth)))
        W = readArrayFromFile(self.estimator.outputDir + "/W/W_clustered.onmf")
        W = np.nan_to_num(W, nan = 1E-14)
        
        for i in range(int(self.blockStructure[0])):
            for j in range(int(self.blockStructure[1])):
                HFilename = self.estimator.outputDir + "/H/H_block_" + str(i * int(self.blockStructure[1]) + j) + ".onmf"
                HBlock = readArrayFromFile(HFilename)
                HBlock = np.nan_to_num(HBlock, nan = 1E-14)
                DBlock = self.estimator.model.computeDensityMap(W, HBlock).reshape((int(self.estimator.blockHeight), int(self.estimator.blockWidth)))
                
                D[i * int(self.estimator.blockHeight) : (i + 1) * int(self.estimator.blockHeight), j * int(self.estimator.blockWidth) : (j + 1) * int(self.estimator.blockWidth)] = DBlock

        return D
    
    
    def computeMeanAtomicNumberMaps(self) -> np.ndarray:
        meanZ = np.zeros((int(self.blockStructure[0] * self.estimator.blockHeight), int(self.blockStructure[1] * self.estimator.blockWidth)))
        W = readArrayFromFile(self.estimator.outputDir + "/W/W_clustered.onmf")
        W = np.nan_to_num(W, nan = 1E-14)
        
        for i in range(int(self.blockStructure[0])):
            for j in range(int(self.blockStructure[1])):
                HFilename = self.estimator.outputDir + "/H/H_block_" + str(i * int(self.blockStructure[1]) + j) + ".onmf"
                HBlock = readArrayFromFile(HFilename)
                HBlock = np.nan_to_num(HBlock, nan = 1E-14)
                meanZBlock = self.estimator.model.computeMeanAtomicNumberMap(W, HBlock).reshape((int(self.estimator.blockHeight), int(self.estimator.blockWidth)))
                
                meanZ[i * int(self.estimator.blockHeight) : (i + 1) * int(self.estimator.blockHeight), j * int(self.estimator.blockWidth) : (j + 1) * int(self.estimator.blockWidth)] = meanZBlock

        return meanZ
    
    
    def computeDensityMap(self) -> np.ndarray:
        D = np.zeros((int(self.blockStructure[0] * self.estimator.blockHeight), int(self.blockStructure[1] * self.estimator.blockWidth)))
        W = readArrayFromFile(self.estimator.outputDir + "/W/W_clustered.onmf")
        W = np.nan_to_num(W, nan = 1E-14)
        H = readArrayFromFile(self.estimator.outputDir + "/H/H_monolithic.onmf")
        H = np.nan_to_num(H, nan = 1E-14)
        D = self.estimator.model.computeDensityMap(W, H).reshape((int(self.blockStructure[0] * self.estimator.blockHeight), int(self.blockStructure[1] * self.estimator.blockWidth)))
        
        return D
    
    
    def computeMeanAtomicNumberMap(self) -> np.ndarray:
        meanZ = np.zeros((int(self.blockStructure[0] * self.estimator.blockHeight), int(self.blockStructure[1] * self.estimator.blockWidth)))
        W = readArrayFromFile(self.estimator.outputDir + "/W/W_clustered.onmf")
        W = np.nan_to_num(W, nan = 1E-14)
        H = readArrayFromFile(self.estimator.outputDir + "/H/H_monolithic.onmf")
        H = np.nan_to_num(H, nan = 1E-14)
        meanZ = self.estimator.model.computeMeanAtomicNumberMap(W, H).reshape((int(self.blockStructure[0] * self.estimator.blockHeight), int(self.blockStructure[1] * self.estimator.blockWidth)))
        
        return meanZ
    
    
    def optimiseQuantificationMaps(
        self, 
        HAADF : hs.signals.Signal2D,
        gamma : np.float64 = 1.6, 
        lambdaHAADF : np.float64 = None, 
        lambdaChem : np.float64 = 0.08, 
        lambdaTV : np.float64 = 0.15, 
        epsilon : np.float64 = 0.24, 
        nIter : int = 30, 
        nIterTV : int = 3, 
        regularise : bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        pathZ = os.path.join(self.estimator.inputDir, "Z")
        pathQOptimised = os.path.join(self.estimator.outputDir, "Q_optimised")
        pathZOptimised = os.path.join(self.estimator.outputDir, "Z_optimised")
        os.makedirs(pathZ, exist_ok = True)
        os.makedirs(pathQOptimised, exist_ok = True)
        os.makedirs(pathZOptimised, exist_ok = True)
        
        HAADF.rechunk((int(self.estimator.blockHeight), int(self.estimator.blockWidth)))
        
        if (lambdaHAADF is None):
            lambdaHAADF = 1 / int(len(list(self.estimator.model.elements)))
        
        gamma = np.float64(gamma)
        lambdaHAADF = np.float64(lambdaHAADF)
        lambdaChem = np.float64(lambdaChem)
        lambdaTV = np.float64(lambdaTV)
        epsilon = np.float64(epsilon)
        nIter = int(nIter)
        nIterTV = int(nIterTV)
        regularise = bool(regularise)
        
        costsHAADF = np.zeros((self.blocks, nIter), dtype = np.float64)
        costsChem = np.zeros((self.blocks, nIter), dtype = np.float64)
        costsTV = np.zeros((self.blocks, nIter), dtype = np.float64)
        
        for i in range(self.blockStructure[0]):
            for j in range(self.blockStructure[1]):
                XInitBlock = HAADF.data.blocks[i, j].compute()
                XInitBlock = np.ascontiguousarray(XInitBlock.reshape((XInitBlock.shape[0] * XInitBlock.shape[1])), dtype = np.float64)
                QFilename = self.estimator.outputDir + "/Q/Q_block_" + str(i * int(self.blockStructure[1]) + j) + ".onmf"
                QBlock = readArrayFromFile(QFilename)
                
                costHAADF = np.ascontiguousarray(np.zeros(shape = (nIter), dtype = np.float64))
                costChem = np.ascontiguousarray(np.zeros(shape = (nIter), dtype = np.float64))
                costTV = np.ascontiguousarray(np.zeros(shape = (nIter), dtype = np.float64))
                
                fusion = core.HAADFDataset(int(self.estimator.blockHeight), int(self.estimator.blockWidth), XInitBlock)
                fusion.loadEDXSQuantificationData(QBlock, self.estimator.model.quantificationElements, self.estimator.model.periodicTableInfoFilePath)
                fusion.runQuantificationDataOptimisationRoutine(gamma, lambdaHAADF, lambdaChem, lambdaTV, epsilon, nIter, nIterTV, costHAADF, costChem, costTV, regularise)
                
                QOptimisedFilename = pathQOptimised + "/Q_block_" + str(i * int(self.blockStructure[1]) + j) + ".onmf"
                ZOptimisedFilename = pathZOptimised + "/Z_block_" + str(i * int(self.blockStructure[1]) + j) + ".onmf"
                writeArrayToFile(QOptimisedFilename, fusion.Q)
                writeArrayToFile(ZOptimisedFilename, fusion.X)
                
                costsHAADF[i * int(self.blockStructure[1]) + j, :] = costHAADF
                costsChem[i * int(self.blockStructure[1]) + j, :] = costChem
                costsTV[i * int(self.blockStructure[1]) + j, :] = costTV
        
        return (costsHAADF, costsChem, costsTV)
    
    
    def optimiseQuantificationMap(
        self, 
        HAADF : hs.signals.Signal2D,
        gamma : np.float64 = 1.6, 
        lambdaHAADF : np.float64 = None, 
        lambdaChem : np.float64 = 0.08, 
        lambdaTV : np.float64 = 0.15, 
        epsilon : np.float64 = 0.24, 
        nIter : int = 30, 
        nIterTV : int = 3, 
        regularise : bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        pathZ = os.path.join(self.estimator.inputDir, "Z")
        pathQOptimised = os.path.join(self.estimator.outputDir, "Q_optimised")
        pathZOptimised = os.path.join(self.estimator.outputDir, "Z_optimised")
        os.makedirs(pathZ, exist_ok = True)
        os.makedirs(pathQOptimised, exist_ok = True)
        os.makedirs(pathZOptimised, exist_ok = True)
        
        if (lambdaHAADF is None):
            lambdaHAADF = 1 / int(len(list(self.estimator.model.elements)))
        
        gamma = np.float64(gamma)
        lambdaHAADF = np.float64(lambdaHAADF)
        lambdaChem = np.float64(lambdaChem)
        lambdaTV = np.float64(lambdaTV)
        epsilon = np.float64(epsilon)
        nIter = int(nIter)
        nIterTV = int(nIterTV)
        regularise = bool(regularise)
        
        XInit = np.ascontiguousarray(HAADF.data.reshape((HAADF.data.shape[0] * HAADF.data.shape[1])), dtype = np.float64)
        QFilename = self.estimator.outputDir + "/Q/Q_monolithic.onmf"
        Q = readArrayFromFile(QFilename)
                
        costHAADF = np.ascontiguousarray(np.zeros(shape = (nIter), dtype = np.float64))
        costChem = np.ascontiguousarray(np.zeros(shape = (nIter), dtype = np.float64))
        costTV = np.ascontiguousarray(np.zeros(shape = (nIter), dtype = np.float64))
        
        fusion = core.HAADFDataset(int(self.estimator.blockHeight), int(self.estimator.blockWidth), XInit)
        fusion.loadEDXSQuantificationData(Q, self.estimator.model.quantificationElements, self.estimator.model.periodicTableInfoFilePath)
        fusion.runQuantificationDataOptimisationRoutine(gamma, lambdaHAADF, lambdaChem, lambdaTV, epsilon, nIter, nIterTV, costHAADF, costChem, costTV, regularise)
        
        QOptimisedFilename = pathQOptimised + "/Z_monolithic.onmf"
        ZOptimisedFilename = pathZOptimised + "/Z_monolithic.onmf"
        writeArrayToFile(QOptimisedFilename, fusion.Q)
        writeArrayToFile(ZOptimisedFilename, fusion.X)
        
        return (costHAADF, costChem, costTV)


    def computeThicknessMaps(self, LL : exspy.signals.LazyEELSSpectrum, D : np.ndarray, ZLP_threshold : np.float64, useOpenMP : Optional[bool] = True, numThreads : Optional[str] = None) -> None:
        LL.rechunk((int(self.estimator.blockHeight), int(self.estimator.blockWidth), LL.data.shape[2]))
        
        energyAxis = np.ascontiguousarray(LL.axes_manager.signal_axes[0].axis, dtype = np.float64)
        energyAxisScale = np.float64(LL.axes_manager.signal_axes[0].scale)
        energyAxisOffset = np.float64(LL.axes_manager.signal_axes[0].offset)
        beamEnergy = np.float64(LL.metadata.Acquisition_instrument.TEM.beam_energy)
        alpha = np.float64(LL.metadata.Acquisition_instrument.TEM.convergence_angle)
        beta = np.float64(LL.metadata.Acquisition_instrument.TEM.Detector.EELS.collection_angle)
        ZLP_threshold = np.float64(ZLP_threshold)
        
        if (useOpenMP):
            if (numThreads is None):
                numThreads = str(os.cpu_count())
            
            os.environ['OMP_NUM_THREADS'] = numThreads
        
        for i in range(LL.data.blocks.shape[0]):
            for j in range(LL.data.blocks.shape[1]):
                LLBlock = LL.data.blocks[i, j, 0].compute()
                LLBlock = LLBlock.reshape((LLBlock.shape[0] * LLBlock.shape[1], LLBlock.shape[2]))
                LLBlock = LLBlock.T
                DBlock = D[i * int(self.estimator.blockHeight) : (i + 1) * int(self.estimator.blockHeight), j * int(self.estimator.blockWidth) : (j + 1) * int(self.estimator.blockWidth)]
                DBlock = DBlock.reshape((DBlock.shape[0] * DBlock.shape[1]))
                EELS = core.EELSDataset(LLBlock, energyAxis, energyAxisScale, energyAxisOffset, beamEnergy, DBlock, alpha, beta, ZLP_threshold)
                EELS.computeThicknessMap()
                TFilename = self.estimator.inputDir + "/T/T_block_" + str(i * int(self.blockStructure[1]) + j) + ".inmf"
                writeArrayToFile(TFilename, EELS.T)
    
    
    def computeThicknessMap(self, LL : exspy.signals.EELSSpectrum, D : np.ndarray, ZLP_threshold : np.float64, useOpenMP : Optional[bool] = True, numThreads : Optional[str] = None) -> None:       
        energyAxis = np.ascontiguousarray(LL.axes_manager.signal_axes[0].axis, dtype = np.float64)
        energyAxisScale = np.float64(LL.axes_manager.signal_axes[0].scale)
        energyAxisOffset = np.float64(LL.axes_manager.signal_axes[0].offset)
        beamEnergy = np.float64(LL.metadata.Acquisition_instrument.TEM.beam_energy)
        alpha = np.float64(LL.metadata.Acquisition_instrument.TEM.convergence_angle)
        beta = np.float64(LL.metadata.Acquisition_instrument.TEM.Detector.EELS.collection_angle)
        ZLP_threshold = np.float64(ZLP_threshold)
        
        if (useOpenMP):
            if (numThreads is None):
                numThreads = str(os.cpu_count())
            
            os.environ['OMP_NUM_THREADS'] = numThreads
            
        LLMatrix = LL.data.reshape((LL.data.shape[0] * LL.data.shape[1], LL.data.shape[2]))
        LLMatrix = LLMatrix.T
        D = D.reshape((D.shape[0] * D.shape[1]))
        EELS = core.EELSDataset(LLMatrix, energyAxis, energyAxisScale, energyAxisOffset, beamEnergy, D, alpha, beta, ZLP_threshold)
        EELS.computeThicknessMap()
        TFilename = self.estimator.inputDir + "/T/T_monolithic.inmf"
        writeArrayToFile(TFilename, EELS.T)
        
        
    def readTBlocks(self) -> np.ndarray:
        T = np.zeros((int(self.blockStructure[0] * self.estimator.blockHeight), int(self.blockStructure[1] * self.estimator.blockWidth)))
        
        for i in range(int(self.blockStructure[0])):
            for j in range(int(self.blockStructure[1])):
                TFilename = self.estimator.inputDir + "/T/T_block_" + str(i * int(self.blockStructure[1]) + j) + ".inmf"
                temp_array = readArrayFromFile(TFilename)
                temp_array = temp_array.reshape((int(self.estimator.blockHeight), int(self.estimator.blockWidth)))
                T[i * int(self.estimator.blockHeight) : (i + 1) * int(self.estimator.blockHeight), j * int(self.estimator.blockWidth) : (j + 1) * int(self.estimator.blockWidth)] = temp_array
                
        return T
    
    
    def computeAbsorptionCorrectionMatrices(self, fusionType : str, useOpenMP : Optional[bool] = True, numThreads : Optional[str] = None) -> None:
        if (useOpenMP):
            if (numThreads is None):
                numThreads = str(os.cpu_count())
            
            os.environ['OMP_NUM_THREADS'] = numThreads
            
        if (fusionType == "EELS"):
            self.estimator.computeABlocks(core.EDXSModelConstants_fusionType.EELS)
        elif (fusionType == "HAADF"):
            self.estimator.computeABlocks(core.EDXSModelConstants_fusionType.HAADF)
        else :
            raise ValueError("Invalid fusion type.")
    
    
    def computeAbsorptionCorrectionMatrix(self, fusionType : str, useOpenMP : Optional[bool] = True, numThreads : Optional[str] = None) -> None:
        if (useOpenMP):
            if (numThreads is None):
                numThreads = str(os.cpu_count())
            
            os.environ['OMP_NUM_THREADS'] = numThreads
        
        if (fusionType == "EELS"):
            self.estimator.computeAMatrix()
        elif (fusionType == "HAADF"):
            self.estimator.computeAMatrix(core.EDXSModelConstants_fusionType.HAADF)
        else :
            raise ValueError("Invalid fusion type.")
    
    
    def readABlocks(self) -> np.ndarray:
        A = da.zeros((self.estimator.model.energyAxisSize, int(self.estimator.blockHeight * self.blockStructure[0]), int(self.estimator.blockWidth * self.blockStrucutre[1])))
        
        for i in range(int(self.blockStructure[0])):
            for j in range(int(self.blockStructure[1])):
                ABlock = readArrayFromFile(self.estimator.outputDir + "/A/A_block_" + str(i * self.blockStructure[1] + j) + ".onmf")
                ABlock = ABlock.reshape((self.estimator.model.energyAxisSize, self.estimator.blockHeight, self.estimator.blockWidth))
                A[:, i * self.estimator.blockHeight : (i + 1) * self.estimator.blockHeight, j * self.estimator.blockWidth : (j + 1) * self.estimator.blockWidth] = ABlock
                
        A = A.transpose(1, 2, 0)
        ASignal = exspy.signals.EDSTEMSpectrum(A, lazy = True)
                
        return ASignal
    
    
    def calculateResiduals(self, nSelectedComponents : Optional[int] = None) -> exspy.signals.LazyEDSTEMSpectrum:
        if (nSelectedComponents is None):
            nSelectedComponents = int(self.estimator.nClusters)
            
        X_block_path = self.estimator.inputDir + "/X/X_block_"
        G_path = self.estimator.outputDir + "/G.onmf"
        W_path = self.estimator.outputDir + "/W/W_clustered.onmf"
        H_block_path = self.estimator.outputDir + "/H/H_block_"
        
        G = readArrayFromFile(G_path)
        W = readArrayFromFile(W_path)
        GW = np.matmul(G, W[:, :nSelectedComponents])
        
        nChannels = G.shape[0]
        
        R = da.zeros((nChannels, int(self.estimator.blockHeight * self.blockStructure[0]), int(self.estimator.blockWidth * self.blockStructure[1])))
        
        for i in range(int(self.blockStructure[0])):
            for j in range(int(self.blockStructure[1])):
                X_block = readArrayFromFile(X_block_path + str(i * int(self.blockStructure[1]) + j) + ".inmf")
                H_block = readArrayFromFile(H_block_path + str(i * int(self.blockStructure[1]) + j) + ".onmf")
                               
                X_R = np.matmul(GW, H_block[:nSelectedComponents, :])
                X_R = X_R / np.linalg.norm(X_R)
                X_block = X_block / np.linalg.norm(X_block)
                
                R_block = X_block - X_R
                R_block = R_block.reshape((nChannels, int(self.estimator.blockHeight), int(self.estimator.blockWidth)))
                R[:, i * int(self.estimator.blockHeight) : (i + 1) * int(self.estimator.blockHeight), j * int(self.estimator.blockWidth) : (j + 1) * int(self.estimator.blockWidth)] = R_block
                
        R[R <= 0.0] = 1e-14
        RSignal = exspy.signals.LazyEDSTEMSpectrum(R.transpose(1, 2, 0), lazy = True)
        
        R = R.rechunk((int(self.estimator.blockHeight), int(self.estimator.blockWidth), R.shape[2]))
        RSignal.rechunk((int(self.estimator.blockHeight), int(self.estimator.blockWidth), RSignal.data.shape[2]))
                
        return RSignal
    
    
    def calculateResidual(self, nSelectedComponents : Optional[int] = None) -> exspy.signals.EDSTEMSpectrum:
        if (nSelectedComponents is None):
            nSelectedComponents = int(self.estimator.nClusters)
            
        X = self.estimator.getMonoliticX()
        G = readArrayFromFile(self.estimator.outputDir + "/G.onmf")
        W = readArrayFromFile(self.estimator.outputDir + "/W/W_clustered.onmf")
        H = readArrayFromFile(self.estimator.outputDir + "/H/H_monolithic.onmf")
        
        GW = np.matmul(G, W[:, :nSelectedComponents])       
        R = np.zeros((X.shape[0], X.shape[1]))
        
        X_R = np.matmul(GW, H[:nSelectedComponents, :])
        X_R = X_R / np.linalg.norm(X_R)
        X = X / np.linalg.norm(X)
        
        R = X - X_R
        R[R <= 0.0] = 1e-14
        R = R.reshape((X.shape[0], int(self.estimator.blockHeight * self.blockStructure[0]), int(self.estimator.blockWidth * self.blockStructure[1])))
        RSignal = exspy.signals.EDSTEMSpectrum(R.transpose(1, 2, 0))
        
        return RSignal
        
        
    def getMonolithicXSignal(self) -> exspy.signals.EDSTEMSpectrum:
        if (self.spatialComputationApproach == "BLOCKWISE"):       
            X = da.zeros((int(self.estimator.model.energyAxisSize), int(self.estimator.blockHeight * self.blockStructure[0]), int(self.estimator.blockWidth * self.blockStructure[1])))
            
            for i in range(int(self.blockStructure[0])):
                for j in range(int(self.blockStructure[1])):
                    XBlock = readArrayFromFile(self.estimator.inputDir + "/X/X_block_" + str(i * int(self.blockStructure[1]) + j) + ".inmf")
                    XBlock = XBlock.reshape((int(self.estimator.model.energyAxisSize), int(self.estimator.blockHeight), int(self.estimator.blockWidth)))
                    X[:, i * int(self.estimator.blockHeight) : (i + 1) * int(self.estimator.blockHeight), j * int(self.estimator.blockWidth) : (j + 1) * int(self.estimator.blockWidth)] = XBlock
                    
            X = X.transpose(1, 2, 0)
            XSignal = exspy.signals.LazyEDSTEMSpectrum(X, lazy = True)
            
            X = X.rechunk((int(self.estimator.blockHeight), int(self.estimator.blockWidth), X.shape[2]))
            XSignal.rechunk((int(self.estimator.blockHeight), int(self.estimator.blockWidth), XSignal.data.shape[2]))
            
        elif (self.spatialComputationApproach == "MONOLITHIC"):
            X = self.estimator.getMonoliticX()
            X = X.reshape((X.shape[0], int(self.estimator.blockHeight * self.blockStructure[0]), int(self.estimator.blockWidth * self.blockStructure[1])))
            X = X.transpose(1, 2, 0)
            XSignal = exspy.signals.EDSTEMSpectrum(X)
                 
        return XSignal
    
    
    def writeAbsorptionCorrectedXBlocks(self) -> None:
        pathNewX = os.path.join(self.estimator.inputDir, "X_corrected")
        os.makedirs(pathNewX, exist_ok = True)
        
        for i in range(int(self.blockStructure[0])):
            for j in range(int(self.blockStructure[1])):
                XBlock = readArrayFromFile(self.estimator.inputDir + "/X/X_block_" + str(i * int(self.blockStructure[1]) + j) + ".inmf")
                ABlock = readArrayFromFile(self.estimator.outputDir + "/A/A_block_" + str(i * int(self.blockStructure[1]) + j) + ".onmf")
                XBlock = XBlock / ABlock
                XNewFilename = self.estimator.inputDir + "/X_corrected/X_corrected_block_" + str(i * int(self.blockStructure[1]) + j) + ".inmf"
                writeArrayToFile(XNewFilename, XBlock)
                              
                
    def getMonolithicAbsorptionCorrectedXSignal(self) -> exspy.signals.EDSTEMSpectrum:
        if (self.spatialComputationApproach == "BLOCKWISE"):
            self.writeAbsorptionCorrectedXBlocks()
            
            X = da.zeros((int(self.estimator.model.energyAxisSize), int(self.estimator.blockHeight * self.blockStructure[0]), int(self.estimator.blockWidth * self.blockStrucutre[1])))
            
            for i in range(int(self.blockStructure[0])):
                for j in range(int(self.blockStructure[1])):
                    XBlock = readArrayFromFile(self.estimator.inputDir + "/X_corrected/X_corrected_block_" + str(i * int(self.blockStructure[1]) + j) + ".inmf")
                    XBlock = XBlock.reshape((int(self.estimator.model.energyAxisSize), int(self.estimator.blockHeight), int(self.estimator.blockWidth)))
                    X[:, i * int(self.estimator.blockHeight) : (i + 1) * int(self.estimator.blockHeight), j * int(self.estimator.blockWidth) : (j + 1) * int(self.estimator.blockWidth)] = XBlock
                    
            X = X.transpose(1, 2, 0)
            XSignal = exspy.signals.LazyEDSTEMSpectrum(X, lazy = True)
            
            X = X.rechunk((int(self.estimator.blockHeight), int(self.estimator.blockWidth), X.shape[2]))
            XSignal.rechunk((int(self.estimator.blockHeight), int(self.estimator.blockWidth), XSignal.data.shape[2]))
            
        elif (self.spatialComputationApproach == "MONOLITHIC"):
            X = self.estimator.getMonoliticX()
            A = readArrayFromFile(self.estimator.outputDir + "/A/A_monolithic.onmf")
            X = X / A
            X = X.reshape((X.shape[0], int(self.estimator.blockHeight * self.blockStructure[0]), int(self.estimator.blockWidth * self.blockStructure[1])))
            X = X.transpose(1, 2, 0)
            XSignal = exspy.signals.EDSTEMSpectrum(X)
                
        return XSignal
    

    def printConcentrationReport(self, fitError : Optional[bool] = True):
        G = self.estimator.G
        W = self.estimator.WClustered
        
        if (self.spatialComputationApproach == "BLOCKWISE"):
            H = self.readHBlocks()
            H = H.reshape((H.shape[0], H.shape[1] * H.shape[2]))
        
        elif (self.spatialComputationApproach == "MONOLITHIC"):
            H = readArrayFromFile(self.estimator.outputDir + "/H/H_monolithic.onmf")
            
        core.printConcentrationReport(G, W, H, self.estimator.model.modelElements, self.estimator.model.quantificationElements, fitError)
        
        