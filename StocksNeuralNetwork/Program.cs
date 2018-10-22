using System;
using System.Collections.Generic;

namespace StocksNeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.Title = "Stocks Neural Network";
            Console.WriteLine("CONFIGURATION"); //Configure program
            float[] input = { };
            float[] target = { };
            bool training = false;
            string path = null;
            int[] layers = { 1, 10, 10, 1 };
            int popSize = 50;

            Console.WriteLine("Training? (y/n)");
            String userTraining = Console.ReadLine();
            if (userTraining == "n") { training = false; } else { training = true; }

            if (!training) { Console.WriteLine("Enter path to neural net"); path = Console.ReadLine(); }

            Console.WriteLine("Population size? (should be even, or will be 20)");
            String userPop = Console.ReadLine();
            int s;
            bool isNumeric1 = int.TryParse(userPop, out s);
            if (isNumeric1) { popSize = int.Parse(userPop); } else { Console.WriteLine("Invalid input, using 50"); }

            Console.WriteLine("How many inputs?");
            String inputCount = Console.ReadLine();
            int n;
            bool isNumeric = int.TryParse(inputCount, out n);
            if (isNumeric) { layers[0] = int.Parse(inputCount); } else { Console.WriteLine("Invalid input, using 1"); }

            Console.WriteLine("How many outputs?");
            String outputCount = Console.ReadLine();
            int o;
            bool isNumeric2 = int.TryParse(outputCount, out o);
            if (isNumeric2) { layers[3] = int.Parse(outputCount); } else { Console.WriteLine("Invalid input, using 1"); }

            for (int i = 0; i < layers[0]; i++) //Fix this
            {
                Console.WriteLine("Enter input " + i);
                String newInput = Console.ReadLine();
                float p;
                bool isNumeric3 = float.TryParse(newInput, out p);
                if (isNumeric3)
                {
                    input[i] = float.Parse(newInput);
                    Console.WriteLine("Updated input " + i);
                }
                else
                {
                    Console.WriteLine("Input invalid, leaving as null");
                }
            }
            for (int i = 0; i < layers[layers.Length - 1]; i++) //Fix this
            {
                Console.WriteLine("Enter target " + i);
                String newInput = Console.ReadLine();
                float q;
                bool isNumeric4 = float.TryParse(newInput, out q);
                if (isNumeric4)
                {
                    target[i] = float.Parse(newInput);
                    Console.WriteLine("Updated target " + i);
                }
                else
                {
                    Console.WriteLine("Input invalid, leaving as null");
                }
            }

            Console.WriteLine("Starting..." + System.Environment.NewLine);
            new Manager(input, target, training, @path, layers, popSize);
            Console.ReadLine();
        }
    }

    public class Manager
    {
        //Variables for managing networks
        private bool isTraning = false;
        private int populationSize = 50;
        private int generationNumber = 0;
        private int[] layers = new int[] { 1, 10, 10, 1 }; //1 input and 1 output
        private List<NeuralNetwork> nets;
        private float[] inputs;
        private float[] targets;

        //Entry point, configure inputs and call update
        public Manager(float[] inData, float[] trainingTargets, bool training, String networkFilePath, int[] layersIn, int PopSize)
        {
            inputs = inData;
            targets = trainingTargets;
            populationSize = PopSize;
            layers = layersIn;
            if (training)
            {
                Update();
            }
            else
            {
                //Use preexising network
                Console.WriteLine();
                Console.WriteLine("Using pre-existing network at " + networkFilePath);
                Console.WriteLine("Layers should be the same, if not, you will encounter issues");
                NeuralNetwork net = new NeuralNetwork(layers);
                net.loadNetwork(networkFilePath);
                Console.WriteLine("Loaded network");
                while (true)
                {
                    float[] output = net.FeedForward(inputs);
                    for (int i = 0; i < output.Length; i++)
                    {
                        Console.WriteLine("Output " + i + ": " + output[i]);
                    }
                    Console.ReadLine();
                    Console.WriteLine("Running input changer...");
                    changeInputTarget();
                }
            }
        }

        void Update()
        {
            if (isTraning == false)
            {
                if (generationNumber == 0) //Perform first time creation if its gen 0
                {
                    InitNeuralNetworks();
                }
                else
                {
                    calculateFitness();
                    nets.Sort();
                    printNetValues();
                    for (int i = 0; i < populationSize / 2; i++) //Loop through the population, mutating the best ones
                    {
                        nets[i] = new NeuralNetwork(nets[i + (populationSize / 2)]);
                        nets[i].Mutate();
                        nets[i + (populationSize / 2)] = new NeuralNetwork(nets[i + (populationSize / 2)]);
                    }

                    for (int i = 0; i < populationSize; i++) //Reset fitness
                    {
                        nets[i].SetFitness(0f);
                    }
                }

                //Increase genertation
                generationNumber++;
                Console.WriteLine();
                if (generationNumber == 1 || generationNumber % 100 == 0)
                {
                    Console.Write("[S]ave best, [E]volve all, [C]hange or check inputs and targets => ");
                    String input = Console.ReadLine().ToLower();
                    if (input == "c" )
                    {
                        changeInputTarget();
                    }
                    else if (input == "s")
                    {
                        Console.WriteLine();
                        Console.WriteLine("Enter file path to save to");
                        String path = Console.ReadLine();
                        nets[populationSize - 1].saveNetwork(path);
                    }
                }
                Update();
            }
        }

        private void changeInputTarget() //Change input and target variables
        {
            Console.WriteLine();
            Console.WriteLine("To change neuron counts, do so in the programs source. It cannot be hot-swapped");
            Console.WriteLine("Current input values:");
            for (int i = 0; i < layers[0]; i++)
            {
                Console.WriteLine("Input " + i + ": " + inputs[i]);
            }
            Console.WriteLine("Current target values:");
            for (int i = 0; i < layers[layers.Length - 1]; i++)
            {
                Console.WriteLine("Target " + i + ": " + targets[i]);
            }
            for (int i = 0; i < layers[0]; i++)
            {
                Console.WriteLine(System.Environment.NewLine + "Enter input " + i);
                String newInput = Console.ReadLine();
                float n;
                bool isNumeric = float.TryParse(newInput, out n);
                if (isNumeric)
                {
                    inputs[i] = float.Parse(newInput);
                    Console.WriteLine("Updated input " + i);
                }
                else
                {
                    Console.WriteLine("Input invalid, leaving as original");
                }
            }
            for (int i = 0; i < layers[layers.Length - 1]; i++)
            {
                Console.WriteLine(System.Environment.NewLine + "Enter target " + i);
                String newInput = Console.ReadLine();
                float n;
                bool isNumeric = float.TryParse(newInput, out n);
                if (isNumeric)
                {
                    targets[i] = float.Parse(newInput);
                    Console.WriteLine("Updated target " + i);
                }
                else
                {
                    Console.WriteLine("Input invalid, leaving as original");
                }
            }
        }

        void calculateFitness() //Calculate fitness of networks
        {
            for (int i = 0; i < populationSize; i++)
            {
                float[] output = nets[i].FeedForward(inputs);
                float fit = 0;
                for (int j = 0; j < output.Length; j++)
                {
                    float actual = (float)output[j];
                    float target = (float)targets[j];
                    float dif = target - actual;
                    float abs = Math.Abs(dif);
                    float fitToAdd = 1 - abs;
                    fit += fitToAdd;
                }
                //Console.WriteLine(fit);
                nets[i].SetFitness(fit);
            }
        }

        //Feed forward and print result
        void printNetValues()
        {
            Console.WriteLine("Generation " + generationNumber);
            float[] output = nets[populationSize - 1].FeedForward(inputs);
            Console.WriteLine("Current best network values: ");
            for (int j = 0; j < output.Length; j++)
            {
                Console.WriteLine("Output " + j + ": " + output[j]);
            }
            Console.WriteLine("Fitness: " + nets[populationSize - 1].GetFitness());
        }

        private void InitNeuralNetworks() //First time setup
        {
            //population must be even, just setting it to 20 incase it's not
            if (populationSize % 2 != 0)
            {
                populationSize = 20;
            }

            nets = new List<NeuralNetwork>(); //Init nets list


            for (int i = 0; i < populationSize; i++) //Populate nets with neural nets
            {
                NeuralNetwork net = new NeuralNetwork(layers);
                net.Mutate();
                nets.Add(net);
            }
        }
    }

    public class NeuralNetwork : IComparable<NeuralNetwork>
    {
        //Neural Network critical variables / data
        private int[] layers;
        private float[][] neurons;
        private float[][][] weights;
        private float fitness;

        public NeuralNetwork(int[] layers)
        {
            //Init layers
            this.layers = new int[layers.Length];
            for (int i = 0; i < layers.Length; i++)
            {
                this.layers[i] = layers[i];
            }

            //Init Matrix
            InitNeurons();
            InitWeights();
        }

        //For copying other networks, literally a copy paste of NeuralNetwork(int[] layers) with copy weights
        public NeuralNetwork(NeuralNetwork copyNetwork)
        {
            this.layers = new int[copyNetwork.layers.Length];
            for (int i = 0; i < copyNetwork.layers.Length; i++)
            {
                this.layers[i] = copyNetwork.layers[i];
            }

            InitNeurons();
            InitWeights();
            CopyWeights(copyNetwork.weights);
        }

        //save the network to file
        public void saveNetwork(string path)
        {
            if (System.IO.File.Exists(path))
            {
                Console.WriteLine("File exists, choose another path");
                return;
            }
            System.IO.File.Create(path).Close();
            for (int i = 0; i < weights.Length; i++)
            {
                for (int j = 0; j < weights[i].Length; j++)
                {
                    for (int k = 0; k < weights[i][j].Length; k++)
                    {
                        using (System.IO.StreamWriter file = new System.IO.StreamWriter(@path, true))
                        {
                            file.WriteLine(weights[i][j][k]);
                        }
                    }
                }
            }
        }

        //save the network to file
        public void loadNetwork(string path)
        {
            if (!System.IO.File.Exists(path))
            {
                Console.WriteLine("File does not exists");
                return;
            }
            for (int i = 0; i < weights.Length; i++)
            {
                for (int j = 0; j < weights[i].Length; j++)
                {
                    for (int k = 0; k < weights[i][j].Length; k++)
                    {
                        int counter = 0;
                        string line;
                        System.IO.StreamReader file = new System.IO.StreamReader(@path);
                        while ((line = file.ReadLine()) != null)
                        {
                            weights[i][j][k] = float.Parse(line);
                            counter++;
                        }
                        file.Close();
                    }
                }
            }
        }

        //When copying other networks, merge its weights with this
        private void CopyWeights(float[][][] copyWeights)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                for (int j = 0; j < weights[i].Length; j++)
                {
                    for (int k = 0; k < weights[i][j].Length; k++)
                    {
                        weights[i][j][k] = copyWeights[i][j][k];
                    }
                }
            }
        }

        private void InitNeurons()
        {
            //Neuron Initilization
            List<float[]> neuronsList = new List<float[]>();
            for (int i = 0; i < layers.Length; i++) //run through all layers
            {
                neuronsList.Add(new float[layers[i]]); //add layer to neuron list
            }
            neurons = neuronsList.ToArray(); //convert list to array
        }

        //Create weights matrix.
        private void InitWeights()
        {
            List<float[][]> weightsList = new List<float[][]>(); //weights list which will later will converted into a weights 3D array

            //itterate over all neurons that have a weight connection
            for (int i = 1; i < layers.Length; i++)
            {
                List<float[]> layerWeightsList = new List<float[]>(); //layer weight list for this current layer (will be converted to 2D array)
                int neuronsInPreviousLayer = layers[i - 1];

                //itterate over all neurons in this current layer
                for (int j = 0; j < neurons[i].Length; j++)
                {
                    float[] neuronWeights = new float[neuronsInPreviousLayer]; //neruons weights

                    //itterate over all neurons in the previous layer and set the weights randomly between 0.5f and -0.5
                    for (int k = 0; k < neuronsInPreviousLayer; k++)
                    {
                        //give random weights to neuron weights
                        neuronWeights[k] = GetRandomNumber(-0.5f, 0.5f);
                    }
                    layerWeightsList.Add(neuronWeights); //add neuron weights of this current layer to layer weights
                }
                weightsList.Add(layerWeightsList.ToArray()); //add this layers weights converted into 2D array into weights list
            }
            weights = weightsList.ToArray(); //convert to 3D array
        }

        /// Feed forward this neural network with a given input array
        public float[] FeedForward(float[] inputs)
        {
            //Add inputs to the neuron matrix
            for (int i = 0; i < inputs.Length; i++)
            {
                neurons[0][i] = inputs[i];
            }

            //itterate over all neurons and compute feedforward values 
            for (int i = 1; i < layers.Length; i++)
            {
                for (int j = 0; j < neurons[i].Length; j++)
                {
                    float value = 0.25f;

                    for (int k = 0; k < neurons[i - 1].Length; k++)
                    {
                        value += weights[i - 1][j][k] * neurons[i - 1][k]; //sum off all weights connections of this neuron weight their values in previous layer
                    }

                    neurons[i][j] = (float)Math.Tanh(value); //Hyperbolic tangent activation
                }
            }

            return neurons[neurons.Length - 1]; //return output layer
        }

        /// Mutate neural network weights
        public void Mutate()
        {
            for (int i = 0; i < weights.Length; i++)
            {
                for (int j = 0; j < weights[i].Length; j++)
                {
                    for (int k = 0; k < weights[i][j].Length; k++)
                    {
                        float weight = weights[i][j][k];

                        //mutate weight value 
                        float randomNumber = GetRandomNumber(0f, 100f);

                        if (randomNumber <= 2f)
                        { //if 1
                          //flip sign of weight
                            weight *= -1f;
                        }
                        else if (randomNumber <= 4f)
                        { //if 2
                          //pick random weight between -1 and 1
                            weight = GetRandomNumber(-0.5f, 0.5f);
                        }
                        else if (randomNumber <= 6f)
                        { //if 3
                          //randomly increase by 0% to 100%
                            float factor = GetRandomNumber(0f, 1f) + 1f;
                            weight *= factor;
                        }
                        else if (randomNumber <= 8f)
                        { //if 4
                          //randomly decrease by 0% to 100%
                            float factor = GetRandomNumber(0f, 1f);
                            weight *= factor;
                        }

                        weights[i][j][k] = weight;
                    }
                }
            }
        }

        //Add fitness to this network
        public void AddFitness(float fit)
        {
            fitness += fit;
        }

        //Set the fitness of this network
        public void SetFitness(float fit)
        {
            fitness = fit;
        }

        //Get the fitness of this network
        public float GetFitness()
        {
            return fitness;
        }

        /// Compare two neural networks and sort based on fitness
        public int CompareTo(NeuralNetwork other)
        {
            if (other == null) return 1;

            if (fitness > other.fitness)
                return 1;
            else if (fitness < other.fitness)
                return -1;
            else
                return 0;
        }

        //Generate a random float
        public float GetRandomNumber(float minimum, float maximum)
        {
            Random random = new Random();
            return (float)random.NextDouble() * (maximum - minimum) + minimum;
        }
    }
}
