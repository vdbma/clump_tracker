use pyo3::prelude::*;

/// A Python module implemented in Rust. The name of this module must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
mod _core {
    use std::cmp::min;
    use std::string::String;
    use pyo3::prelude::*;
    use pyo3::types::{PyDict,PyInt,PyList,PyFloat,PyString};
    use numpy::ndarray::{Array,ArrayRef, Array1, Array2,ArrayD, ArrayView1,ArrayView3, ArrayViewD, ArrayViewMutD,meshgrid, MeshIndex, Axis, Zip,Slice,IxDyn};
    use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArrayDyn, PyArrayMethods};
    use num_traits::{Float, Signed};
    use std::ops::{Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign,Neg};

    trait AtLeastF32: Float + From<f32> + Signed +numpy::ndarray::ScalarOperand
    + Copy
    + Clone
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Neg<Output = Self>
{
}
    impl AtLeastF32 for f32 {}
    impl AtLeastF32 for f64 {}

    fn compute_adjacency_cartesian<T:AtLeastF32>(indexes:&Vec<Vec<usize>>,
        x:&ArrayView1<'_, T>,
        y:&ArrayView1<'_, T>,
        z:&ArrayView1<'_, T>,
        d: T ) -> Array2<u8> {
        // indexes : list of 3D indexes
        // d : distance to be considered connected

        let mut adj = Array2::<u8>::zeros((indexes.len(),indexes.len())); // no bool arra :()

        for i in 0..indexes.len() {
            for j in 0..indexes.len() {
                let d2 = (x[indexes[i][0]]-x[indexes[j][0]])*(x[indexes[i][0]]-x[indexes[j][0]])
                        +(y[indexes[i][1]]-y[indexes[j][1]])*(y[indexes[i][1]]-y[indexes[j][1]])
                        +(z[indexes[i][2]]-z[indexes[j][2]])*(z[indexes[i][2]]-z[indexes[j][2]])  ;
                adj[[i,j]] = (d2 <= d*d).try_into().unwrap();
            }
        }
        adj
    }

    #[pyfunction(name="compute_adjacency_cartesian_f32")]
    fn compute_adjacency_cartesian_f32_py<'py>( py: Python<'py>,
                            indexes:Bound<'_, PyList>,
                            x: &Bound<'py, PyArray1<f32>>,
                            y: &Bound<'py, PyArray1<f32>>,
                            z: &Bound<'py, PyArray1<f32>>,
                            d: Bound<'_, PyFloat>
                        ) -> Bound<'py, PyArray2<u8>> {
        let r_x = unsafe {x.as_array()};
        let r_y = unsafe {y.as_array()};
        let r_z = unsafe {z.as_array()};
        let r_d:f32 = d.extract().unwrap();
        let r_indexes:Vec<Vec<usize>> = indexes.extract().unwrap();

        compute_adjacency_cartesian::<f32>(&r_indexes,&r_x,&r_y,&r_z,r_d).into_pyarray(py)
    }

    #[pyfunction(name="compute_adjacency_cartesian_f64")]
    fn compute_adjacency_cartesian_f64_py<'py>( py: Python<'py>,
                            indexes:Bound<'_, PyList>,
                            x: &Bound<'py, PyArray1<f64>>,
                            y: &Bound<'py, PyArray1<f64>>,
                            z: &Bound<'py, PyArray1<f64>>,
                            d: Bound<'_, PyFloat>
                        ) -> Bound<'py, PyArray2<u8>> {
        let r_x = unsafe {x.as_array()};
        let r_y = unsafe {y.as_array()};
        let r_z = unsafe {z.as_array()};
        let r_d:f64 = d.extract().unwrap();
        let r_indexes:Vec<Vec<usize>> = indexes.extract().unwrap();

        compute_adjacency_cartesian::<f64>(&r_indexes,&r_x,&r_y,&r_z,r_d).into_pyarray(py)
    }

    fn compute_cc<T:AtLeastF32>(indexes:&Vec<Vec<usize>>,
        x:&ArrayView1<'_, T>,
        y:&ArrayView1<'_, T>,
        z:&ArrayView1<'_, T>,
        d: T,
        geometry:String ) -> Vec<Vec<usize>> {


    let mut composante_connexes:Vec<Vec<usize>> = Vec::new();
    let mut nb_cc:usize = 0;

    let mut deja_vus = Array1::<u8>::zeros(indexes.len());

    let adj;
    if geometry == "cartesian" {
            adj = compute_adjacency_cartesian::<T>(indexes,x,y,z,d);
    } else {
        unimplemented!()
    }


    for (i,_) in indexes.iter().enumerate(){
        let mut a_visiter = Array1::<u8>::zeros(indexes.len());
        for k in 0..indexes.len() {
            a_visiter[k] = adj[[i,k]];
        }
        a_visiter[[i]] = 0;
        deja_vus[[i]] = 1;

        for k in 0..indexes.len() {
            if a_visiter[k] == 1 && deja_vus[k] == 0 {
                a_visiter[k] = 1;
            } else {
                a_visiter[k] = 0; // useless
            }
        }

        composante_connexes.push([i].to_vec());
        nb_cc +=1;

        while a_visiter.sum()>0 {
            for (j,c) in indexes.iter().enumerate() {
                if deja_vus[j] == 0 {
                    for k in 0..indexes.len() {
                        a_visiter[k] += adj[[i,k]];
                        a_visiter[k] = min(a_visiter[k],1);
                    }
                    a_visiter[j] = 0;
                    deja_vus[j] = 1;

                    for k in 0..indexes.len() {
                        if a_visiter[k] == 1 && deja_vus[k] == 0 {
                            a_visiter[k] = 1;
                        } else {
                            a_visiter[k] = 0;
                        }
                    }
                    composante_connexes[nb_cc-1].push(j);
                }
            }
        }

        let nb_deja_vus:usize = deja_vus.sum().try_into().unwrap();
        if  nb_deja_vus== indexes.len(){
            break;
        }


    }

    composante_connexes
    }

    #[pyfunction(name="compute_cc_f32")]
    fn compute_cc_f32_py<'py>( py: Python<'py>,
                            indexes:Bound<'_, PyList>,
                            x: &Bound<'py, PyArray1<f32>>,
                            y: &Bound<'py, PyArray1<f32>>,
                            z: &Bound<'py, PyArray1<f32>>,
                            d: Bound<'_, PyFloat>,
                            geometry: Bound<'_, PyString>
                        ) -> Bound<'py, PyList> {
        let r_x = unsafe {x.as_array()};
        let r_y = unsafe {y.as_array()};
        let r_z = unsafe {z.as_array()};
        let r_d:f32 = d.extract().unwrap();
        let r_indexes:Vec<Vec<usize>> = indexes.extract().unwrap();
        let r_geometry:String = geometry.extract().unwrap();

        PyList::new(py,compute_cc::<f32>(&r_indexes,&r_x,&r_y,&r_z,r_d,r_geometry)).expect("???")
    }

    #[pyfunction(name="compute_cc_f64")]
    fn compute_cc_f64_py<'py>( py: Python<'py>,
                            indexes:Bound<'_, PyList>,
                            x: &Bound<'py, PyArray1<f64>>,
                            y: &Bound<'py, PyArray1<f64>>,
                            z: &Bound<'py, PyArray1<f64>>,
                            d: Bound<'_, PyFloat>,
                            geometry: Bound<'_, PyString>
                        ) -> Bound<'py, PyList> {
        let r_x = unsafe {x.as_array()};
        let r_y = unsafe {y.as_array()};
        let r_z = unsafe {z.as_array()};
        let r_d:f64 = d.extract().unwrap();
        let r_indexes:Vec<Vec<usize>> = indexes.extract().unwrap();
        let r_geometry:String = geometry.extract().unwrap();

        PyList::new(py,compute_cc::<f64>(&r_indexes,&r_x,&r_y,&r_z,r_d,r_geometry)).expect("???")
    }
}
