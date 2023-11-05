//
//  ViewController.swift
//  ObjectDetection-CoreML
//
//  Created by Julius Hietala on 16.8.2022.
//

import AVFoundation
import UIKit
import AVFoundation
import Vision

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    // Capture
    var bufferSize: CGSize = .zero
    var inferenceTime: CFTimeInterval  = 0;
    private let session = AVCaptureSession()
    var audioPlayer: AVAudioPlayer?
    var isAlertSoundPlaying = false
    var redTintView: UIView?
    
    
    // UI/Layers
    @IBOutlet weak var previewView: UIView!
    var rootLayer: CALayer! = nil
    private var previewLayer: AVCaptureVideoPreviewLayer! = nil
    private var detectionLayer: CALayer! = nil
    private var inferenceTimeLayer: CALayer! = nil
    private var inferenceTimeBounds: CGRect! = nil
    
    // Vision
    private var requests = [VNRequest]()
    
    // Setup
    override func viewDidLoad() {
        super.viewDidLoad()
        setupCapture()
        setupOutput()
        setupLayers()
        try? setupVision()
        session.startRunning()
    }
    
    func playSound(named soundFileName: String) {
        // If the sound is already playing, don't try to play it again
        guard !isAlertSoundPlaying else { return }

        guard let url = Bundle.main.url(forResource: soundFileName, withExtension: "wav") else {
            print("Sound file named \(soundFileName) not found.")
            return
        }

        do {
            audioPlayer = try AVAudioPlayer(contentsOf: url)
            audioPlayer?.delegate = self // Set the delegate to self
            isAlertSoundPlaying = true // Set the flag to true
            audioPlayer?.play()
        } catch {
            print("Couldn’t load the sound file named \(soundFileName): \(error)")
            isAlertSoundPlaying = false // Reset the flag if the sound couldn't be played
        }
    }

    
    func setupCapture() {
        var deviceInput: AVCaptureDeviceInput!
        let videoDevice = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera], mediaType: .video, position: .back).devices.first
        do {
            deviceInput = try AVCaptureDeviceInput(device: videoDevice!)
        } catch {
            print("Could not create video device input: \(error)")
            return
        }
        
        session.beginConfiguration()
        session.sessionPreset = .vga640x480
        
        guard session.canAddInput(deviceInput) else {
            print("Could not add video device input to the session")
            session.commitConfiguration()
            return
        }
        session.addInput(deviceInput)
        
        do {
            try  videoDevice!.lockForConfiguration()
            let dimensions = CMVideoFormatDescriptionGetDimensions((videoDevice?.activeFormat.formatDescription)!)
            bufferSize.width = CGFloat(dimensions.width)
            bufferSize.height = CGFloat(dimensions.height)
            videoDevice!.unlockForConfiguration()
        } catch {
            print(error)
        }
        session.commitConfiguration()
    }
    
    func setupOutput() {
        let videoDataOutput = AVCaptureVideoDataOutput()
        let videoDataOutputQueue = DispatchQueue(label: "VideoDataOutput", qos: .userInitiated, attributes: [], autoreleaseFrequency: .workItem)
        
        if session.canAddOutput(videoDataOutput) {
            session.addOutput(videoDataOutput)
            videoDataOutput.alwaysDiscardsLateVideoFrames = true
            videoDataOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)]
            videoDataOutput.setSampleBufferDelegate(self, queue: videoDataOutputQueue)
        } else {
            print("Could not add video data output to the session")
            session.commitConfiguration()
            return
        }
    }
    
    func setupLayers() {
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill
        rootLayer = previewView.layer
        previewLayer.frame = rootLayer.bounds
        rootLayer.addSublayer(previewLayer)
        
        inferenceTimeBounds = CGRect(x: rootLayer.frame.midX-75, y: rootLayer.frame.maxY-70, width: 150, height: 17)
        
        inferenceTimeLayer = createRectLayer(inferenceTimeBounds, [1,1,1,1])
        inferenceTimeLayer.cornerRadius = 7
        rootLayer.addSublayer(inferenceTimeLayer)
        
        redTintView = UIView(frame: rootLayer.bounds)
        redTintView?.backgroundColor = UIColor.red.withAlphaComponent(0.5) // Semi-transparent
        redTintView?.isHidden = true // Hidden by default
        redTintView?.isUserInteractionEnabled = false // Disable user interaction
        rootLayer.addSublayer(redTintView!.layer)
        
        detectionLayer = CALayer()
        detectionLayer.bounds = CGRect(x: 0.0,
                                       y: 0.0,
                                       width: bufferSize.width,
                                       height: bufferSize.height)
        detectionLayer.position = CGPoint(x: rootLayer.bounds.midX, y: rootLayer.bounds.midY)
        rootLayer.addSublayer(detectionLayer)
        
        let xScale: CGFloat = rootLayer.bounds.size.width / bufferSize.height
        let yScale: CGFloat = rootLayer.bounds.size.height / bufferSize.width
        
        let scale = fmax(xScale, yScale)
        
        // rotate the layer into screen orientation and scale and mirror
        detectionLayer.setAffineTransform(CGAffineTransform(rotationAngle: CGFloat(.pi / 2.0)).scaledBy(x: scale, y: -scale))
        // center the layer
        detectionLayer.position = CGPoint(x: rootLayer.bounds.midX, y: rootLayer.bounds.midY)
    }
    
    func setupVision() throws {
        guard let modelURL = Bundle.main.url(forResource: "yolov5n", withExtension: "mlmodelc") else {
            throw NSError(domain: "ViewController", code: -1, userInfo: [NSLocalizedDescriptionKey: "Model file is missing"])
        }
        
        do {
            let visionModel = try VNCoreMLModel(for: MLModel(contentsOf: modelURL))
            let objectRecognition = VNCoreMLRequest(model: visionModel, completionHandler: { (request, error) in
                DispatchQueue.main.async(execute: {
                    if let results = request.results {
                        self.drawResults(results)
                    }
                })
            })
            self.requests = [objectRecognition]
        } catch let error as NSError {
            print("Model loading went wrong: \(error)")
        }
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])
        do {
            // returns true when complete https://developer.apple.com/documentation/vision/vnimagerequesthandler/2880297-perform
            let start = CACurrentMediaTime()
            try imageRequestHandler.perform(self.requests)
            inferenceTime = (CACurrentMediaTime() - start)
            
        } catch {
            print(error)
        }
    }
    
    func drawResults(_ results: [Any]) {
        CATransaction.begin()
        CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions)
        detectionLayer.sublayers = nil // Clear previous detections from detectionLayer
        inferenceTimeLayer.sublayers = nil
        for observation in results where observation is VNRecognizedObjectObservation {
            guard let objectObservation = observation as? VNRecognizedObjectObservation else {
                continue
            }
            
            // Detection with highest confidence
            guard let topLabelObservation = objectObservation.labels.first else { continue }
            
            // Rotate the bounding box into screen orientation
            let boundingBox = CGRect(origin: CGPoint(x: 1.0 - objectObservation.boundingBox.origin.y - objectObservation.boundingBox.size.height,
                                                     y: objectObservation.boundingBox.origin.x),
                                     size: CGSize(width: objectObservation.boundingBox.size.height,
                                                  height: objectObservation.boundingBox.size.width))
            
            let objectBounds = VNImageRectForNormalizedRect(boundingBox, Int(bufferSize.width), Int(bufferSize.height))
            
            // Get color for the identifier, or use a default color if not found
            let color = colors[topLabelObservation.identifier] ?? UIColor.red.cgColor as! [CGFloat] // Default to red if color not found
            
            let shapeLayer = createRectLayer(objectBounds, color)
            
            // Calculate the size of the bounding box in the appropriate scale
            let width = objectBounds.size.width
            let height = objectBounds.size.height
            let sizeString = String(format: "%.2f x %.2f", width, height)
            
            // Combine label, confidence, and size information
            let labelString = String(format: "%@\n%.1f%%\nSize: %@", topLabelObservation.identifier.capitalized, topLabelObservation.confidence * 100, sizeString)
            let formattedString = NSMutableAttributedString(string: labelString)
            //if(formattedString.string.contains("Traffic")) {
                let textLayer = createDetectionTextLayer(objectBounds, formattedString)
                shapeLayer.addSublayer(textLayer)
                detectionLayer.addSublayer(shapeLayer)
            //}
            
            let objectArea = objectBounds.width * objectBounds.height
            let screenArea = bufferSize.width * bufferSize.height
            if objectArea / screenArea > 0.15 {
                DispatchQueue.main.async {
                    self.redTintView?.isHidden = false
                    self.playSound(named: "alert")
                    // Hide the red tint view after a second
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
                        self.redTintView?.isHidden = true
                    }
                }
            }
            
            let label = topLabelObservation.identifier.lowercased()
                    if label.contains("traffic light") { // Assuming "traffic light" is the identifier used in the model
                        playSound(named: "red")
                    }
            
            let formattedInferenceTimeString = NSMutableAttributedString(string: String(format: "Inference time: %.1f ms", inferenceTime * 1000))
            
            let inferenceTimeTextLayer = createInferenceTimeTextLayer(inferenceTimeBounds, formattedInferenceTimeString)
            inferenceTimeLayer.addSublayer(inferenceTimeTextLayer)
            
            CATransaction.commit()
        }
        
        // Clean up capture setup
        func teardownAVCapture() {
            previewLayer.removeFromSuperlayer()
            previewLayer = nil
        }
        
    }
}

extension ViewController: AVAudioPlayerDelegate {
    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        // When the audio player finishes playing, reset the flag
        isAlertSoundPlaying = false
    }

    func audioPlayerDecodeErrorDidOccur(_ player: AVAudioPlayer, error: Error?) {
        // If there is a decode error, also reset the flag
        print("Audio player decode error: \(String(describing: error))")
        isAlertSoundPlaying = false
    }
}

