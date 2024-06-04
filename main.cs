using System;
using System.IO;
using Microsoft.Data.Analysis;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.CompilerServices;
using System.Collections;
using System.Linq;




class Program
{
    static void Main(string[] args)
    {
      
        // Точка входа в программу
        var df = GetPath();
        // Console.WriteLine(df);
        var result_tuple = ModelTrain(df);
        // Console.WriteLine(result_tuple);
        var train_data = result_tuple.Item2;
        var forecast_data = result_tuple.Item1;

        var forecast_lower = result_tuple.Item3;
        var forecast_upper = result_tuple.Item4;


        PlotTest(train_data, forecast_data, forecast_lower,forecast_upper);
        Console.WriteLine("Скрипт отработал успешно");
    }


    static (string, string) GetPath()
    {
    
        var dataPathTrain = Path.GetFullPath(@"train.csv");
        var dataPathTest = Path.GetFullPath(@"test.csv");

        return (dataPathTrain, dataPathTest);
    }

    static void PlotTest(double[] train_data, float[] forecast_data, float[] forecast_lower,float[] forecast_upper)
    {
      
        int[] numbers_train = Enumerable.Range(1, 52).ToArray();
        int[] numbers_forecast = Enumerable.Range(53, forecast_data.Length).ToArray();
        int[] numbers_forecast_lower = Enumerable.Range(53, forecast_lower.Length).ToArray();
        int[] numbers_forecast_upper = Enumerable.Range(53, forecast_upper.Length).ToArray();
        
        
        ScottPlot.Plot myPlot = new();
        var train_plot = myPlot.Add.Scatter(numbers_train, train_data);
        var forecast_plot = myPlot.Add.Scatter(numbers_forecast, forecast_data);
        var forecast_lower_plot = myPlot.Add.Scatter(numbers_forecast_lower, forecast_lower);
        var forecast_upper_plot = myPlot.Add.Scatter(numbers_forecast_upper, forecast_upper);

        train_plot.LegendText = "Train";
        forecast_plot.LegendText = "Forecast";
        forecast_lower_plot.LegendText = "lower";
        forecast_upper_plot.LegendText = "upper";

        myPlot.Title("Forecast C# ");
        myPlot.XLabel("Время");
        myPlot.YLabel("SALES VOL");
        myPlot.ShowLegend();
        myPlot.SavePng("plot.png", 800, 600);


    }



    static (float[],double[],float[],float[]) ModelTrain((string, string) df)
    {
        string path_train = df.Item1;
        string path_test = df.Item2;



        MLContext mlContext = new MLContext();

        IDataView data1View =
            mlContext.Data.LoadFromTextFile<ModelInput>(path_train, separatorChar: ',', hasHeader: false);
        IDataView data2View =
            mlContext.Data.LoadFromTextFile<ModelInput>(path_test, separatorChar: ',', hasHeader: false);


        var forecastingPipeline = mlContext.Forecasting.ForecastBySsa(
            outputColumnName: "forecasted_count",
            inputColumnName: "count",
            windowSize: 8,
            seriesLength: 30,
            trainSize: 52,
            horizon: 50,
            confidenceLevel: 0.95f,
            confidenceLowerBoundColumn: "lower_count",
            confidenceUpperBoundColumn: "upper_count");
      

        SsaForecastingTransformer forecaster = forecastingPipeline.Fit(data1View);
      
        var forecastEngine = forecaster.CreateTimeSeriesEngine<ModelInput, ModelOutput>(mlContext);
        
        ModelOutput forecast = forecastEngine.Predict();
        

        // Считываем все строки из файла
        string[] lines = File.ReadAllLines(path_train);
        // Создаем массивы для хранения данных
        double[] floatArray = new double[lines.Length];
        
        // Перебираем строки CSV-файла и разбираем значения

        for (int i = 1; i < lines.Length; i++)
        {   
            
            
            string modifiedString_1 = lines[i].Replace(',', ';');
            
            string modifiedString_2 = modifiedString_1.Replace('.', ',');
          
            string[] values = modifiedString_2.Split(';');
           
            // Парсим значения в типы float и int
            var slice_str = values[0].Substring(0, 6);

            double floatValue = double.Parse(slice_str); // Первый столбец в C
        
            // Сохраняем значения в массивы
            
            floatArray[i] = floatValue;

        }

        double[] slicedArray = floatArray.Skip(1).ToArray();
        
        return (forecast.forecasted_count,slicedArray, forecast.lower_count,forecast.upper_count);

    }
}




public class ModelInput
{
    [LoadColumn(1)] public DateTime action_time { get; set; }
    [LoadColumn(0)] public float count { get; set; }
}

public class TrainData
{
    [LoadColumn(1)] public DateTime action_time { get; set; }

    [LoadColumn(0)] public float count { get; set; }



}

public class ModelOutput
{
    public float[] forecasted_count { get; set; }
    public float[] lower_count { get; set; }
    public float[] upper_count { get; set; }
}

