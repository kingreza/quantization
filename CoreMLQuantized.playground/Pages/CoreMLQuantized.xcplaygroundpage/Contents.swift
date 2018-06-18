import Vision
import CoreML
import Cocoa

let testingFolder = "/Users/rezashirazian/Projects/Practice/Quantize/testing_data/"

let modelFolder = "/Users/rezashirazian/Projects/Practice/Quantize/"

func getCIImage(url: URL) -> CIImage {
    guard let image = NSImage(contentsOf: url) else {
        fatalError()
    }
    let data = image.tiffRepresentation!
    let bitmap = NSBitmapImageRep(data: data)!
    let ciimage = CIImage(bitmapImageRep: bitmap)!
    return ciimage
}

func getFoldersInDirectory(path: String) -> [String:URL]  {
    guard let contents = try? FileManager.default.contentsOfDirectory(atPath: path) else {
        print("Make sure you have set a correct value for testingFolder and modelFolder variables at the top of this playground.")
        fatalError()
    }
    
    let contentURLs = contents.map{URL(fileURLWithPath: path + $0)}
    let folders = contentURLs.filter{$0.hasDirectoryPath}
    var result: [String:URL] = [:]
    folders.forEach{
        result[$0.lastPathComponent.split(separator: "_").joined(separator: " ")] = $0.absoluteURL
    }
    return result
}

func getFilesInFolder(url: URL) -> [URL] {
    guard let contents = try? FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil, options: .skipsHiddenFiles) else {
        print("Make sure you have set a correct value for modelFolder variables at the top of this playground and that there are models present.")
        fatalError()
    }
    return contents.filter{!$0.hasDirectoryPath}
}

func getImagesFromFolder(url: URL) -> [CIImage] {
    
    let contents = getFilesInFolder(url: url)
    let images = contents.map {getCIImage(url:$0)}
    
    return images
}

func getModelsFromFolder(url: URL) -> [(MLModel, String, Int)] {
    let contents = getFilesInFolder(url: url).filter{$0.absoluteString.hasSuffix(".mlmodel")}
    //var result : [(MLModel, String, Int)] = []
    var result : [(MLModel, String, Int)] = []
    contents.forEach {
        //print("adding model: \($0.absoluteString)")
        let fileSize = (try! FileManager.default.attributesOfItem(atPath: $0.path)[FileAttributeKey.size] as! NSNumber).intValue
        let name = $0.lastPathComponent
        let model = try! MLModel(contentsOf: MLModel.compileModel(at: $0))
        result.append((model, name, fileSize))
    }
    return result
}

func calculateAccuracy(images:[CIImage], label: String, model: MLModel) -> Double {
    guard images.count > 0 else {
        print("Make sure you have set a correct value for testingFolder variables at the top of this playground and that there are test images present.")
        fatalError()
    }
    var matched = 0
    
    do {
        let visionModel = try VNCoreMLModel(for: model)
        let visionRequest = VNCoreMLRequest(model: visionModel) { (request, error) in
            let result = request.results as? [VNClassificationObservation]
            if let matchLabel = result?.first?.identifier {
                //print("was hoping for  \(label) got  \(matchLabel)")
                if matchLabel == label {
                    matched += 1
                }
            }
        }
        let handler = VNSequenceRequestHandler()
        for image in images {
            try handler.perform([visionRequest], on: image)
        }
        
    } catch (let error) {
        print (error)
    }
    
    return Double(matched) / Double(images.count)
}

func calculateAccuracy(images: [String: [CIImage]], model: MLModel) -> [String: Double] {
    guard images.values.count > 0 else {
        fatalError()
    }
    
    var result: [String: Double] = [:]
    
    images.keys.forEach { key in
        if let images = images[key] {
            result[key] = calculateAccuracy(images: images, label: key, model: model)
        }
    }
    return result
}

let folders = getFoldersInDirectory(path: testingFolder)

var images: [String: [CIImage]] = [:]

folders.keys.forEach {
    if let value = folders[$0] {
        images[$0] = getImagesFromFolder(url: value)
    }
}

let models = getModelsFromFolder(url: URL(fileURLWithPath: modelFolder)).sorted { left, right  in
    return left.1 < right.1
}

print("report for \(modelFolder)")
print("total of \(models.count) models")
var csv = "model, size,"
var firstLine = true
for (model, name, size) in models {
    print("\tmodel:\t\(name)")
    print("\tsize:\t\(size)")
    let accuracyReports = calculateAccuracy(images: images, model: model)
    if(firstLine) {
        csv += accuracyReports.keys.sorted().joined(separator: ",") + "\n"
        firstLine = false
    }
    csv += "\(name),\(size),"
    for report in accuracyReports.keys.sorted() {
        print("\t\t\(report):\t\t\t\(accuracyReports[report]!)")
        csv += "\(accuracyReports[report]!),"
    }
    csv += "\n"
    print("\n")
}
let destination = URL(fileURLWithPath: modelFolder + "result.csv")
try! csv.write(to: destination, atomically: true, encoding: String.Encoding.utf8)


